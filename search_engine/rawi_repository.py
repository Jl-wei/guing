import os
import json
import requests

class RaWiRepository():
    def __init__(self):
        self.rico_screenshot_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/rico/images")
        self.eval_query_results = json.load(open(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "search_engine/eval/cache/rawi.json")))
    
    def query(self, text, count=10):
        text = text.strip()
        if count <= 10 and (text in self.eval_query_results):
            result = self.eval_query_results[text][:count]
            result = [x.split("/")[-1].split(".")[0] for x in result]
            return [self.image_path(x) for x in result]
        else:
            print("Retrieve from RaWi")
            url = "http://rawi-prototyping.com/gui2r/v1/retrieval"
            payload = {"query": text, "method": "bm25okapi", "qe_method": "", "max_results": count}
            r = requests.post(url, json=payload)
            return [self.image_path(x['index']) for x in r.json()['results']]
    
    def image_path(self, screen_id):
        file_path = os.path.join(self.rico_screenshot_path, f"{screen_id}.jpg")
        return file_path

        # return f"http://rawi-prototyping.com/static/resources/combined/{screen_id}.jpg"
