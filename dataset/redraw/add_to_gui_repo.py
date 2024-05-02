import os

from tqdm import tqdm
from pymongo.mongo_client import MongoClient

if __name__ == "__main__":
    uri = os.environ['MONGODB_URI']
    print(uri)
    client = MongoClient(uri)
    db = client['google_play']
    
    images_path = "./images"
    for app_id in tqdm(os.listdir(images_path)):
        for img_name in os.listdir(os.path.join(images_path, app_id)):
            if img_name.startswith(".") or (not img_name.endswith(".png")): continue

            db['gui_repository'].insert_one({
                "app_id": app_id,
                "image_name": img_name,
                "source": "redraw",
                "type": 'screenshot'
            })
