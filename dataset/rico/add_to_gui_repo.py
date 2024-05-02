import os
import pandas as pd

from tqdm import tqdm
from pymongo.mongo_client import MongoClient

if __name__ == "__main__":
    uri = os.environ['MONGODB_URI']
    print(uri)
    client = MongoClient(uri)
    db = client['google_play']
    
    df = pd.read_csv("./ui_details.csv")
    
    for img_name in tqdm(os.listdir("./images")):
        if not img_name.endswith(".jpg"): continue

        app_id = df[df['UI Number'] == int(img_name.split(".")[0])].iloc[0]['App Package Name']
        db['gui_repository'].insert_one({
            "app_id": app_id,
            "image_name": img_name,
            "source": "rico",
            "type": 'screenshot'
        })
