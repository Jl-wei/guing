import torch
import os
import argparse

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm

from sketch_vit import ViTForSketchMatching
from sketch_vgg import VggForSketchMatching

class SwireDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        sketch = Image.open(item['sketch'])
        screenshot = Image.open(item['screenshot'])
        
        sketch_encoding = self.processor(images=[sketch], return_tensors="pt")
        sketch_encoding = {k: v.squeeze() for k, v in sketch_encoding.items()}
        
        screenshot_encoding = self.processor(images=[screenshot], return_tensors="pt")
        screenshot_encoding = {k: v.squeeze() for k, v in screenshot_encoding.items()}
        
        return {
            "sketch_pixel_values": sketch_encoding['pixel_values'], 
            "screenshot_pixel_values": screenshot_encoding['pixel_values']
        }


def main(model_name, batch_size, lr, n_epoch, save_per_epoch):
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("../../retrieval/models/clip-base-mix-all")
    
    if model_name == "vgg":
        model = VggForSketchMatching()
    elif model_name == "clip-vit":
        model = ViTForSketchMatching()
    elif model_name == "uiclip-vit":
        model = ViTForSketchMatching("../../retrieval/models/clip-base-mix-all/checkpoint-11460")
    else:
        raise Exception("Unknow model")
    
    dataset = load_dataset(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/train-dataset.py"), split="train")
    train_dataset = SwireDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(n_epoch):
        print("Epoch:", epoch)
        pbar = tqdm(train_dataloader)
        for idx, batch in enumerate(pbar):
            sketch_pixel_values = batch.pop("sketch_pixel_values").to(device)
            screenshot_pixel_values = batch.pop("screenshot_pixel_values").to(device)
            
            outputs = model(screen_pixel_values=screenshot_pixel_values,
                            sketch_pixel_values=sketch_pixel_values,)
            
            loss = outputs["loss"]

            pbar.set_description(f"Loss: {round(loss.item(), 5)}, LR: {optimizer.param_groups[0]['lr']}")

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % save_per_epoch == 0:    
            path = f"./models/{model_name}"
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, f"{epoch}.pt"))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-n', '--n_epoch', type=int, default=10)
    parser.add_argument('-s', '--save_per_epoch', type=int, default=1)

    args = parser.parse_args()

    main(args.model_name, args.batch_size, args.learning_rate, args.n_epoch, args.save_per_epoch)

# python train.py -m vgg -b 64 -n 100 -s 10