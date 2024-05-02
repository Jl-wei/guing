import os
import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import classification_report

from linear_probe_utils import enrico_labels, enrico_labels_desc


def predict_img(img_path, label_emb):
    image = Image.open(img_path)
    image = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    img_emb = model.get_image_features(image)
    img_emb = img_emb.detach().cpu().numpy()
    scores = np.dot(img_emb, label_emb.T)
    pred = np.argmax(scores)
    return pred

def compute_label_emb(model, processor, labels):
    label_tokens = processor(text=labels, padding=True, return_tensors='pt').to(device)
    label_emb = model.get_text_features(**label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
    return label_emb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../../retrieval/models/clip-base-mix-all/checkpoint-11460"
    # model_path = "openai/clip-vit-base-patch32"
    
    if model_path.startswith('.') or model_path.startswith('/'):
        processor_path = "/".join(model_path.split('/')[0:-1])
    else:
        processor_path = model_path
    
    model = CLIPModel.from_pretrained(model_path).to(device)    
    processor = CLIPProcessor.from_pretrained(processor_path)
    
    model.to(device)

    label_emb = np.zeros([20, 512])

    _labels = [":".join(label) for label in enrico_labels_desc]
    _label_emb = compute_label_emb(model, processor, _labels)
    label_emb = label_emb + _label_emb

    enrico_dataset = load_dataset(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/dataset.py"), split="train")

    y_pred = []
    y_pred_id = []
    y_true = []
    y_true_id = []
    for i in tqdm(range(len(enrico_dataset))):
        img_path = enrico_dataset[i]['image']
        pred = predict_img(img_path, label_emb)
        y_pred.append(enrico_labels_desc[pred][0].lower())
        y_pred_id.append(pred)
        
        y_true.append(enrico_dataset[i]['topic'].lower())
        y_true_id.append(enrico_labels.index(enrico_dataset[i]['topic']))
    
    # print(classification_report(y_true, y_pred, labels=enrico_labels))
    print(classification_report(y_true_id, y_pred_id, target_names=enrico_labels, digits=3))
