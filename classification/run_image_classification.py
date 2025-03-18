from PIL import Image
from transformers import pipeline
from tqdm import tqdm

import argparse
import os
import jsonlines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--images_path')
    parser.add_argument('-m', '--model_path', default="Jl-wei/app-intro-img-classifier")

    args = parser.parse_args()
    images_path = args.images_path
    model_path = args.model_path

    classifier = pipeline("image-classification", model="model_path", device=0)

    with jsonlines.open("./prediction.jsonl", mode='w') as writer:
        for app_id in tqdm(os.listdir(images_path)):
            for image_name in os.listdir(os.path.join(images_path, app_id)):
                if image_name.endswith(".webp"):
                    img_path = os.path.join(images_path, app_id, image_name)
                    image = Image.open(img_path)
                    result = classifier(image)
                    writer.write({"app_id": app_id, "image_name": image_name, "result": result})
