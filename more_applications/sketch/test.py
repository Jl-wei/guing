import requests
import torch
import jsonlines
import os

from PIL import Image
from transformers import AutoProcessor, CLIPModel
from sentence_transformers import util
from tqdm import tqdm

from sketch_vit import ViTForSketchMatching
from sketch_vgg import VggForSketchMatching


def compute_sketch2screenshot_recall(sim_matrix, k=1):
    inds = torch.argsort(sim_matrix, dim=1, descending=True)
    topk = inds[:, :k]

    num_text = sim_matrix.shape[0]
    ground_truth = torch.arange(num_text).to(device)
    
    correct = torch.eq(topk, ground_truth.unsqueeze(-1)).any(dim=1)
    num_correct = correct.sum().item()
    
    return num_correct / num_text

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # model = ViTForSketchMatching("../../retrieval/models/clip-base-mix-all/checkpoint-11460").to(device)
    # model.load_state_dict(torch.load("./models/uiclip-vit/9.pt"))
    
    # model = ViTForSketchMatching("openai/clip-vit-base-patch32").to(device)
    # model.load_state_dict(torch.load("./models/clip-vit/9.pt"))

    model = VggForSketchMatching().to(device)
    model.load_state_dict(torch.load("./models/vgg/99.pt"))
    model.eval()
    
    processor = AutoProcessor.from_pretrained("../../retrieval/models/clip-base-mix-all")

    sketches = []
    sketches_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/sketches")
    screenshots = []
    screenshots_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/screenshots")
    with jsonlines.open(os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/test-data.jsonl"), mode='r') as reader:
        for item in reader:
            sketches.append(Image.open(os.path.join(sketches_path, item['sketch'])))
            screenshots.append(Image.open(os.path.join(screenshots_path, item['screenshot'])))

    sketch_features = []
    for image in tqdm(sketches):
        image_inputs = processor(images=[image], return_tensors="pt").to(device)
        if type(model) == VggForSketchMatching:
            image_inputs = image_inputs.pixel_values
            sketch_features.append(model.sketch_model(image_inputs).detach())
        else:
            sketch_features.append(model.sketch_model(**image_inputs).image_embeds.detach())
    sketch_features = torch.cat(sketch_features)
    print(sketch_features.shape)

    screenshot_features = []
    for image in tqdm(screenshots):
        image_inputs = processor(images=[image], return_tensors="pt").to(device)
        if type(model) == VggForSketchMatching:
            image_inputs = image_inputs.pixel_values
            screenshot_features.append(model.screen_model(image_inputs).detach())
        else:
            screenshot_features.append(model.screen_model(**image_inputs).image_embeds.detach())
    screenshot_features = torch.cat(screenshot_features)
    print(screenshot_features.shape)

    sim_matrix = util.pytorch_cos_sim(sketch_features, screenshot_features)
    print(sim_matrix)

    for k in [1, 3, 5, 10]:
        value = compute_sketch2screenshot_recall(sim_matrix, k)
        print(f"Sketch to Screenshot Recall@{k}: {value}")
