import torch
from transformers import AutoProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR

from retrieval_util import image_caption_pairs_from_dataset, encode_dataset, recall_at_k


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset_name = "google_play"
    # dataset_name = "screen2words"
    # dataset_name = "clarity"

    model_path = "openai/clip-vit-base-patch32"
    # model_path = "./models/clip-base-mix-all-train/checkpoint-10640"
    
    if model_path.startswith('.') or model_path.startswith('/'):
        processor_path = "/".join(model_path.split('/')[0:-1])
    else:
        processor_path = model_path
    
    clip_model = CLIPModel.from_pretrained(model_path).to(device)    
    clip_processor = AutoProcessor.from_pretrained(processor_path)
    
    text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    ocr = PaddleOCR(lang='en', show_log=False)
    
    images, captions = image_caption_pairs_from_dataset(dataset_name, split="test", length=None)
    
    text_encodings, image_encodings, text_to_image_map = encode_dataset(images, captions, device, clip_model=clip_model, clip_processor=clip_processor)
    # text_encodings, image_encodings, text_to_image_map = encode_dataset(images, captions, device, text_model=text_model, ocr=ocr)
    
    dist_matrix = text_encodings @ image_encodings.T

    print(dataset_name)
    print(model_path)
    for k in [1, 3, 5, 10, int(len(images)/100), int(len(images)/20), int(len(images)/10)]:
        value = recall_at_k(dist_matrix, text_to_image_map, k)
        print(f"T2I Recall@{k}: {value}")
