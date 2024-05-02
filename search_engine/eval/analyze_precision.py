import os
from pymongo.mongo_client import MongoClient

def print_precisions(selected_image_ids, name=""):
    print(name)
    for k in [1, 3, 5, 10]:
        print(f"Precision at {k}:", precision_at_k(selected_image_ids, k))

def precision_at_k(selected_image_ids, k):
    # remove the image ids that are larger than k
    selected_image_ids = list(map(lambda x: list(filter(lambda i: int(i) < k, x)), selected_image_ids))
    precision = sum(list(map(lambda x: len(x), selected_image_ids))) / (k * len(selected_image_ids))
    return precision
    
def print_hits(selected_image_ids, name=""):
    print(name)
    for k in [1, 3, 5, 10]:
        print(f"HITS at {k}:", hits_at_k(selected_image_ids, k))

def hits_at_k(selected_image_ids, k):
    # remove the image ids that are larger than k
    selected_image_ids = list(map(lambda x: list(filter(lambda i: int(i) < k, x)), selected_image_ids))
    hits = sum(list(map(lambda x: 0 if len(x)==0 else 1, selected_image_ids))) / len(selected_image_ids)
    return hits

def mean_reciprocal_rank(selected_image_ids):
    mrr = sum(list(map(lambda x: 0 if len(x)==0 else 1/(int(x[0])+1), selected_image_ids))) / len(selected_image_ids)
    return mrr

if __name__ == "__main__":
    mongo_client = MongoClient(os.environ['MONGODB_URI'])

    results = list(mongo_client['evaluation']["search_engine"].find({"user": {"$ne": "test"}}))

    mix_selected_image_ids = []
    rico_redraw_selected_image_ids = []
    rawi_selected_image_ids = []
    
    for result in results:
        mix_selected_image_ids.append(result["mix_selected_image_ids"][1:])
        rico_redraw_selected_image_ids.append(result["rico_redraw_selected_image_ids"][1:])
        rawi_selected_image_ids.append(result["rawi_selected_image_ids"][1:])
    
    print("Precision")
    print_precisions(mix_selected_image_ids, "mix")
    print_precisions(rico_redraw_selected_image_ids, "rico_redraw")
    print_precisions(rawi_selected_image_ids, "rawi")

    print("HITS")
    print_hits(mix_selected_image_ids, "mix")
    print_hits(rico_redraw_selected_image_ids, "rico_redraw")
    print_hits(rawi_selected_image_ids, "rawi")
    
    print("MRR")
    print("mix", mean_reciprocal_rank(mix_selected_image_ids))
    print("rico_redraw", mean_reciprocal_rank(rico_redraw_selected_image_ids))
    print("rawi", mean_reciprocal_rank(rawi_selected_image_ids))
    