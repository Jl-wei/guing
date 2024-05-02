# GUing: A Mobile GUI Search Engine using a Vision-Language Model

GUing is a GUI search engine based on a large vision-language model called UIClip, which we trained specifically for the app GUI domain.
For this, we first collected app introduction images from Google Play, which usually display the most representative screenshots selected and often captioned (i.e.~labeled) by app vendors.
Then, we developed an automated pipeline to classify, crop, and extract the captions from these images.
This finally results in a large dataset which we share with this paper: including 303k app screenshots, out of which 135k have captions.
We used this dataset to train a novel vision-language model, which is, to the best of our knowledge, the first of its kind in GUI retrieval. 

![image](./assets/approach-overview.png)

If you find our work useful, please cite our paper:
```bibtex
@misc{wei2024guing,
      title={GUing: A Mobile GUI Search Engine using a Vision-Language Model}, 
      author={Jialiang Wei and Anne-Lise Courbis and Thomas Lambolais and Binbin Xu and Pierre Louis Bernard and Gérard Dray and Walid Maalej},
      year={2024},
      eprint={2405.00145},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

## Setup

1. Install `poetry`

2. Install dependencies
```
poetry install
```

3. Set the environment variables
```
export GUI_SEARCH_ROOT_PATH= your project path
export MONGODB_URI= your mongodb uri
```

## UIClip

You can find our `UIClip` model that was pretrained on GPSCap, Screen2Words and Clarity dataset at: https://huggingface.co/Jl-wei/uiclip-vit-base-patch32

### UIClip for GUI retrieval

The code for UIClip training and testing is in the `retrieval` folder.

To train the classifier, use the `train_clip.py` script.

To run the classifier on whole dataset, use the `test.py` script.

### UIClip for image classification

The code is in the `more_applications/classification` folder.

### UIClip for text-to-sketch retrieval

The code is in the `more_applications/sketch` folder.



## Dataset

The labels for the `GPSCap` (Google Play Screenshot Caption) dataset are available at `dataset/google_play/captioning`. Due to their substantial size, we are unable to provide the images directly. However, you can download these images from Google Play using the IDs provided in the `data.jsonl` file, and then crop them according to the specified bounding boxes.

The datasets utilized for training the image classifier (`dataset/google_play/classification`) and the screen cropper (`dataset/google_play/detection`) will be released upon the acceptance of our paper.

You can access the other datasets by downloading them from their respective websites.



## Dataset Creation Pipeline

![image](./assets/dataset-creation.png)

### Image Classification

The code for classification is in the `classification` folder.

To train the classifier, use the `train_image_classification.py` script.

To run the classifier on whole dataset, use the `run_image_classification.py` script.

### Screen Cropping

The code for cropping screens is in the `detection` folder.

To train the detector, use the `train_object_detection.py` script.

To run the detector on whole dataset, use the `run_object_detection.py` script.

### Caption Recognition

The code for caption extraction is in the `ocr` folder.

To run the detector on whole dataset, use the `run_ocr.py` script.



## Search Engine

![image](./assets/guing-overview.png)

The code for `GUing`, our GUI search engine, is in the `search_engine` folder.

You need to add data to MongoDB and create `faiss` index before use it.

