import os
import torch

from PIL import Image
from tqdm import tqdm
from pymongo.mongo_client import MongoClient
from transformers import AutoProcessor, CLIPModel


def encode_image(image, processor, model, device):
    image = Image.open(image_path)
    image_inputs = processor(images=[image], return_tensors="pt").to(device)
    img_emb = model.get_image_features(**image_inputs).detach()
    return img_emb.tolist()[0]

if __name__ == "__main__":
    embedding_field = "embedding"
    
    uri = os.environ['MONGODB_URI']
    print(uri)
    client = MongoClient(uri)
    db = client['google_play']

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all/checkpoint-11460")).to(device)
    processor = AutoProcessor.from_pretrained(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all"))

    gp_surrounded_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/captioning/png_images")
    gp_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/images")
    rico_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/rico/images")
    redraw_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/redraw/images")
    
    cursor = db['gui_repository'].find({embedding_field: {"$exists": False}}, no_cursor_timeout=True).sort('_id', -1).batch_size(1)
    length = db['gui_repository'].count_documents({embedding_field: {"$exists": False}})
    for screen in tqdm(cursor, total=length):
        if screen["source"] == "google_play":
            if screen["type"] == "surrounded_screenshot":
                image_name = f"{screen['image_name'].split('.')[0]}.png"
                image_path = os.path.join(gp_surrounded_screenshot_path, image_name)
            else:
                image_path = os.path.join(gp_screenshot_path, screen['app_id'], screen['image_name'])
        elif screen["source"] == "rico":
            image_path = os.path.join(rico_screenshot_path, screen['image_name'])
        elif screen["source"] == "redraw":
            image_path = os.path.join(redraw_screenshot_path, screen['app_id'], screen['image_name'])
        
        image = Image.open(image_path)
        img_emb = encode_image(image, processor, model, device)
        db['gui_repository'].update_one({"_id": screen['_id']}, {"$set": {embedding_field: img_emb}})
