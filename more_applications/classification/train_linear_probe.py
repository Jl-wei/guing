import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from linear_probe_utils import LinearProbeClassifier, EnricoDataset, enrico_labels


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--repeat', type=int, default=1)
    args = parser.parse_args()
    
    # embedding_method = "text"
    # embedding_method = "clip"
    embedding_method = "uiclip"
    enrico = load_dataset(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/dataset.py"), split="train")
    enrico_dataset = EnricoDataset(enrico, embedding_method)

    for i in range(args.repeat):
        # model = LinearProbeClassifier(embedding_size=1024).to(device)
        model = LinearProbeClassifier(embedding_size=512).to(device)
        
        train_idx, test_idx = train_test_split(np.arange(len(enrico_dataset)),
                                                    test_size=0.2,
                                                    random_state=916,
                                                    shuffle=True,
                                                    stratify=enrico_dataset.img_labels)

        train_dataset = Subset(enrico_dataset, train_idx)
        test_dataset = Subset(enrico_dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        
        for epoch in range(100):
            print("Epoch", epoch)
            pbar = tqdm(train_dataloader)
            for i, data in enumerate(pbar):
                inputs, labels = data['embedding'].to(device), data['label'].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                pbar.set_description(f"Loss: {round(loss.item(), 5)}, LR: {optimizer.param_groups[0]['lr']}")

            if (epoch+1) % 100 == 0:
                path = f"./models/{embedding_method}"
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path, f"{epoch}.pth"))

                y_pred = []
                y_true = []
                with torch.no_grad():
                    for data in test_dataloader:
                        inputs, labels = data['embedding'].to(device), data['label'].to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        y_pred.extend(predicted.tolist())
                        y_true.extend(labels.tolist())

                print(classification_report(y_true, y_pred, target_names=enrico_labels, digits=3))
