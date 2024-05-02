import os
import torch
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
from sklearn.metrics import classification_report

from linear_probe_utils import enrico_labels, enrico_labels_desc

def predict_img(img_path, label_emb, ocr):
    image = cv2.imread(img_path)
    result = ocr.ocr(image, cls=False)
    text = ""
    for item in result[0]:
        text = text + ";" + item[1][0]
    img_emb = model.encode(text)

    scores = np.dot(img_emb, label_emb.T)
    pred = np.argmax(scores)
    return pred

if __name__ == "__main__":
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    ocr = PaddleOCR(lang='en', show_log=False)

    labels = [":".join(label) for label in enrico_labels_desc]
    label_emb = model.encode(labels)

    enrico_dataset = load_dataset(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/dataset.py"), split="train")

    y_pred = []
    y_pred_id = []
    y_true = []
    y_true_id = []
    for i in tqdm(range(len(enrico_dataset))):
        img_path = enrico_dataset[i]['image']
        pred = predict_img(img_path, label_emb, ocr)
        y_pred.append(enrico_labels_desc[pred][0].lower())
        y_pred_id.append(pred)
        
        y_true.append(enrico_dataset[i]['topic'].lower())
        y_true_id.append(enrico_labels.index(enrico_dataset[i]['topic']))
    
    # print(classification_report(y_true, y_pred, labels=enrico_labels))
    print(classification_report(y_true_id, y_pred_id, target_names=enrico_labels, digits=3))
