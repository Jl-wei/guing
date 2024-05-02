import os
import faiss
import numpy as np

from pymongo.mongo_client import MongoClient


if __name__ == "__main__":
    embedding_field = "embedding"
    index_file_name = "gui_repository_mix_11460"
    
    uri = os.environ['MONGODB_URI']
    print(uri)
    client = MongoClient(uri)
    db = client['google_play']
    
    dimension = len(db['gui_repository'].find_one({embedding_field: {"$exists": True}})[embedding_field])
    embeddings = []
    screen_ids = []
    for document in db['gui_repository'].find({embedding_field: {"$exists": True}}):
        embeddings.append(document[embedding_field])
        screen_ids.append(document['screen_id'])
    embeddings = np.array(embeddings)
    screen_ids = np.array(screen_ids)
    
    storage = "Flat"
    n_cells = 3000
    params = f"IVF{n_cells},{storage}"
    index = faiss.index_factory(dimension, params)
    index.nprobe = 1000
    index.train(embeddings)
    index.add_with_ids(embeddings, screen_ids)
    
    faiss.write_index(index, f"{index_file_name}.index")