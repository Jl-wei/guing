import os
import faiss
import torch

from pymongo.mongo_client import MongoClient
from transformers import AutoProcessor, CLIPModel

class GUIRepository():
    def __init__(self, index_path, processor_path, model_path):
        uri = os.environ['MONGODB_URI']
        client = MongoClient(uri)
        self.db = client['google_play']
        self.coll = self.db['gui_repository']
        
        self.index = faiss.read_index(index_path)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(processor_path)
        
        self.gp_surrounded_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/captioning/png_images")
        self.gp_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/images")
        self.rico_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/rico/images")
        self.redraw_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/redraw/images")
        
    def query(self, text, count=10):
        text_inputs = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
        text_embedding = self.model.get_text_features(**text_inputs).detach().cpu()
        prob, ids = self.index.search(text_embedding, count)
        
        prob_ids = list(zip(prob[0].tolist(), ids[0].tolist()))
        prob_ids.sort(reverse=True, key=lambda x: x[0])
        
        paths = []
        for prob_id in prob_ids:
            path = self.image_path(prob_id[1])
            paths.append(path)
        
        return paths
    
    def image_path(self, screen_id):
        screen = self.coll.find_one({"screen_id": screen_id})
        if screen['source'] == "google_play":
            if screen["type"] == "surrounded_screenshot":
                image_name = f"{screen['image_name'].split('.')[0]}.png"
                image_path = os.path.join(self.gp_surrounded_screenshot_path, image_name)
            else:
                image_path = os.path.join(self.gp_screenshot_path, screen['app_id'], screen['image_name'])
        elif screen['source'] == "rico":
            image_path = os.path.join(self.rico_screenshot_path, screen['image_name'])
        elif screen['source'] == "redraw":
            image_path = os.path.join(self.redraw_screenshot_path, screen['image_name'])

        return image_path

if __name__ == "__main__":
    index_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/gui_repository/gui_repository.index")
    model_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-gp/clip-base-mix-all/checkpoint-11460")
    processor_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all")

    db = GUIRepository(index_path=index_path, model_path=model_path, processor_path=processor_path)
    result = db.query("step count")
    print(result)
