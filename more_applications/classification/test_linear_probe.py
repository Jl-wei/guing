import torch
import torch.nn.functional as F
import os

from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

from linear_probe_utils import LinearProbeClassifier, EnricoDataset, enrico_labels


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_method = "uiclip"
    model = LinearProbeClassifier(embedding_size=512).to(device)
    model.load_state_dict(torch.load("./models/uiclip.pth"))
    
    enrico_dataset = load_dataset(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/dataset.py"), split="train")
    test_dataset = EnricoDataset(enrico_dataset, embedding_method)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=128)
    
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data['embedding'].to(device), data['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())

    print(classification_report(y_true, y_pred, target_names=enrico_labels))
