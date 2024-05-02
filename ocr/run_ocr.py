import os
import cv2
import jsonlines
import argparse

from paddleocr import PaddleOCR
from tqdm import tqdm

# python run_ocr.py -i ../dataset/google_play/images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--images_path')

    args = parser.parse_args()
    images_path = args.images_path
    
    ocr = PaddleOCR(lang='en', show_log=False)
    
    with jsonlines.open("./prediction.jsonl", mode='w') as writer:
        for app_name in tqdm(os.listdir(images_path)):
            for filename in os.listdir(os.path.join(images_path, app_name)):
                if filename.endswith(".webp"):
                    img_path = os.path.join(images_path, app_name, filename)
                    image = cv2.imread(img_path)
                    result = ocr.ocr(image, cls=False)
                    writer.write({"app_name": app_name, "filename": filename, "result": result})
