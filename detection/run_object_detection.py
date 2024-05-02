import os
import argparse
import jsonlines

from PIL import Image
from transformers import pipeline
from tqdm import tqdm
from pymongo.mongo_client import MongoClient


if __name__ == "__main__":
    uri = os.environ['MONGODB_URI']
    print(uri)
    client = MongoClient(uri)
    db = client['google_play']
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--images_path')
    parser.add_argument('-m', '--model_path')
    
    args = parser.parse_args()
    images_path = args.images_path
    model_path = args.model_path

    detector = pipeline("object-detection", model=model_path, device=0)
    cls_pred = list(db.classification_pred.find({"result": {"$elemMatch": {"score": {"$gt": 0.5}, "label": 0}}}))
    
    with jsonlines.open("./prediction.jsonl", mode='w') as writer:
        for pred in tqdm(cls_pred):
            app_id = pred['app_id']
            image_name = pred['image_name']
            
            img_path = os.path.join(images_path, app_id, image_name)
            image = Image.open(img_path)
            result = detector(image, threshold=0.5)
            writer.write({"app_id": app_id, "image_name": image_name, "result": result})
