# Modified from https://github.com/openai/CLIP/issues/115

import torch
import jsonlines
import os
import cv2
import random
from tqdm import tqdm


def image_caption_pairs_from_dataset(name, split="test", length=1000):
    '''
    images are a list of images
    captions are a list of list, the inner list are the captions of one image
    
    Args: (`str`)
        split should be "", "test" or "val"
    '''
    
    if name == "screen2words":
        images_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/screen2words/images")
        meta_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], f"dataset/screen2words/{split}-data.jsonl")
    elif name == "clarity":
        images_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/clarity/Clarity-Data/Clarity-PNGs")
        meta_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], f"dataset/clarity/{split}-data.jsonl")
    elif name == "google_play":
        images_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/captioning/png_images")
        meta_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], f"dataset/google_play/captioning/{split}-data.jsonl")

    image_caption_pairs = {}
    with jsonlines.open(meta_path, mode='r') as reader:
        for item in reader:
            if item['image_name'] in image_caption_pairs:
                image_caption_pairs[item['image_name']].append(item['caption'])
            else:
                image_caption_pairs[item['image_name']] = [item['caption']]
    
    captions = []
    images = []
    i = 0
    icp = list(image_caption_pairs.items())
    random.shuffle(icp)
    for key, value in tqdm(icp, total=length):
        image = cv2.imread(os.path.join(images_path, key))
        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_cv)
        captions.append(value)
        # if length is none, prepare all data in the dataset
        if length:
            i += 1
            if i >= length: break
    
    return images, captions

def encode_dataset(images, captions, device, clip_model=None, clip_processor=None, text_model=None, ocr=None):
    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []
    
    image_encodings = []
    text_encodings = []

    assert len(images) == len(captions)

    if clip_model and clip_processor and not text_model and not ocr:
        for i in tqdm(range(len(images))):
            image_inputs = clip_processor(images=[images[i]], return_tensors="pt").to(device)
            image_encodings.append(clip_model.get_image_features(**image_inputs).detach())

            text_inputs = clip_processor(text=captions[i], padding=True, return_tensors="pt").to(device)
            text_encodings.append(clip_model.get_text_features(**text_inputs).detach())

            text_to_image_map += [i] * len(captions[i])

    if not clip_model and not clip_processor and text_model and ocr:
        for i in tqdm(range(len(images))):
            ocr_result = ocr.ocr(images[i], cls=False)
            text = "passage: "
            for item in ocr_result[0]: 
                text = text + ";" + item[1][0]
            image_encodings.append(text_model.encode([text], convert_to_tensor=True))
            text_encodings.append(text_model.encode(list(map(lambda x: "query: " + x, captions[i])), convert_to_tensor=True))

            text_to_image_map += [i] * len(captions[i])

    if clip_model and clip_processor and text_model and ocr:
        for i in tqdm(range(len(images))):
            ocr_result = ocr.ocr(images[i], cls=False)
            text = "passage: "
            for item in ocr_result[0]: 
                text = text + ";" + item[1][0]

            ocr_image_embedding = text_model.encode([text], convert_to_tensor=True)
            ocr_text_embedding = text_model.encode(list(map(lambda x: "query: " + x, captions[i])), convert_to_tensor=True)
            
            image_inputs = clip_processor(images=[images[i]], return_tensors="pt").to(device)
            clip_image_embedding = clip_model.get_image_features(**image_inputs).detach()
            text_inputs = clip_processor(text=captions[i], padding=True, return_tensors="pt").to(device)
            clip_text_embedding = clip_model.get_text_features(**text_inputs).detach()

            image_encodings.append(torch.add(ocr_image_embedding, torch.cat([clip_image_embedding,clip_image_embedding], 1)))
            text_encodings.append(torch.add(ocr_text_embedding, torch.cat([clip_text_embedding,clip_text_embedding],1)))

            text_to_image_map += [i] * len(captions[i])

    image_encodings = torch.cat(image_encodings)
    text_encodings = torch.cat(text_encodings)
    image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)

    return text_encodings, image_encodings, text_to_image_map

def recall_at_k(dist_matrix, text_to_image_map, k):
    num_text = dist_matrix.shape[0]

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)

    # Extract top k indices only
    topk = inds[:, :k]

    # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
    correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
    num_correct = correct.sum().item()
    text_to_image_recall = num_correct / num_text
    
    return text_to_image_recall