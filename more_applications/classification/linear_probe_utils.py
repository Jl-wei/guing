import torch
import cv2

from torch import nn
from torch.utils.data import Dataset
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR

class LinearProbeClassifier(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.linear_probe = nn.Linear(embedding_size, 20)

    def forward(self, embedding):
        return self.linear_probe(embedding)

enrico_labels_desc = [
    ("Bare", "Largely unused area"),
    ("Dialer", "Number entry"),
    ("Camera", "Camera functionality"),
    ("Chat", "Chat functionality"),
    ("Editor", "Text/Image editing"),
    ("Form ", "Form filling functionality"),
    ("Gallery", "Grid-like layout with images"),
    ("List", "Elements organized in a column"),
    ("Login", "Input fields for logging"),
    ("Maps", "Geographic display"),
    ("MediaPlayer", "Music or video player"),
    ("Menu", "Items list in an overlay or aside"),
    ("Modal", "A popup-like window"),
    ("News", "Snippets list: image, title, text"),
    ("Other", "Everything else (rejection class)"),
    ("Profile", "Info on a user profile or product"),
    ("Search", "Search engine functionality"),
    ("Settings", "Controls to change app settings"),
    ("Terms", "Terms and conditions of service"),
    ("Tutorial", "Onboarding screen"),
]

enrico_labels = [
    "bare", 
    "dialer",
    "camera",
    "chat",
    "editor", 
    "form", 
    "gallery",
    "list",
    "login",
    "maps",
    "mediaplayer",
    "menu",
    "modal",
    "news",
    "other",
    "profile",
    "search", 
    "settings",
    "terms",
    "tutorial",
]

class EnricoDataset(Dataset):
    def __init__(self, dataset, embedding_method):
        self.img_labels = []
        for item in dataset:
            label = torch.tensor(enrico_labels.index(item['topic']))
            self.img_labels.append(label)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if embedding_method == "uiclip":
            processor = AutoProcessor.from_pretrained("../../retrieval/models/clip-base-mix-all")
            model = CLIPVisionModelWithProjection.from_pretrained("../../retrieval/models/clip-base-mix-all/checkpoint-11460").to(device)
        elif embedding_method == "clip":
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        elif embedding_method == "text":
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            ocr = PaddleOCR(lang='en', show_log=False)
        
        # embed the entire dataset
        self.processed_dataset = []
        for item in tqdm(dataset):
            label = torch.tensor(enrico_labels.index(item['topic']))
            
            if embedding_method == "uiclip" or embedding_method == "clip":
                image = Image.open(item['image'])
                image_inputs = processor(images=[image], return_tensors="pt").to(device)
                embedding = model(**image_inputs).image_embeds.squeeze().detach()
            elif embedding_method == "text":
                image = cv2.imread(item['image'])
                result = ocr.ocr(image, cls=False)
                text = ""
                for item in result[0]:
                    text = text + ";" + item[1][0]
                embedding = model.encode(text)
            
            self.processed_dataset.append({"embedding": embedding, "label": label})

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[int(idx)]
