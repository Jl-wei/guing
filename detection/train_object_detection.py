import logging
import os
import sys
import warnings
import datetime
import random
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import json
import pandas as pd
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
import torchvision
import albumentations
import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError(
                "You must specify a dataset name."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    num_queries: int = field(
        default=100,
        metadata={"help": "The number of the object queries"},   
    )


def main(random_seed=False):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) <= 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    training_args.output_dir = os.path.join(training_args.output_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if random_seed:
        training_args.seed = random.randrange(1000)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(labels, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        num_queries=model_args.num_queries,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    _train_transforms = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(),
            albumentations.RandomBrightnessContrast(),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )
    
    def formatted_anns(image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations

    def transform_aug_ann(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = _train_transforms(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        # dataset["train"].set_transform(train_transforms)
        dataset["train"] = dataset["train"].with_transform(transform_aug_ann)

    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")

        # format annotations the same as for training, no need for data augmentation
        def val_formatted_anns(image_id, objects):
            annotations = []
            for i in range(0, len(objects["id"])):
                new_ann = {
                    "id": objects["id"][i],
                    "category_id": objects["category"][i],
                    "iscrowd": 0,
                    "image_id": image_id,
                    "area": objects["area"][i],
                    "bbox": objects["bbox"][i],
                }
                annotations.append(new_ann)

            return annotations


        # Save images and annotations into the files torchvision.datasets.CocoDetection expects
        def save_annotation_file_images(examples):
            output_json = {}
            path_output = os.path.join(training_args.output_dir, "evaluation", "examples")

            if not os.path.exists(path_output):
                os.makedirs(path_output)

            path_anno = os.path.join(path_output, "ann.json")
            categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
            output_json["images"] = []
            output_json["annotations"] = []
            for example in examples:
                ann = val_formatted_anns(example["image_id"], example["objects"])
                output_json["images"].append(
                    {
                        "id": example["image_id"],
                        "width": example["image"].width,
                        "height": example["image"].height,
                        "file_name": f"{example['image_id']}.png",
                    }
                )
                output_json["annotations"].extend(ann)
            output_json["categories"] = categories_json

            with open(path_anno, "w") as file:
                json.dump(output_json, file, ensure_ascii=False, indent=4)

            for im, img_id in zip(examples["image"], examples["image_id"]):
                path_img = os.path.join(path_output, f"{img_id}.png")
                im.save(path_img)

            return path_output, path_anno
        
        class CocoDetection(torchvision.datasets.CocoDetection):
            def __init__(self, img_folder, image_processor, ann_file):
                super().__init__(img_folder, ann_file)
                self.image_processor = image_processor

            def __getitem__(self, idx):
                # read in PIL image and target in COCO format
                img, target = super(CocoDetection, self).__getitem__(idx)

                # preprocess image and target: converting target to DETR format,
                # resizing + normalization of both image and target)
                image_id = self.ids[idx]
                target = {"image_id": image_id, "annotations": target}
                encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
                pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
                target = encoding["labels"][0]  # remove batch dimension

                return {"pixel_values": pixel_values, "labels": target}


        im_processor = AutoImageProcessor.from_pretrained(training_args.output_dir)
        path_output, path_anno = save_annotation_file_images(dataset["validation"])
        test_ds_coco_format = CocoDetection(path_output, im_processor, path_anno)

        model = AutoModelForObjectDetection.from_pretrained(training_args.output_dir)
        module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
        val_dataloader = torch.utils.data.DataLoader(
            test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                pixel_values = batch["pixel_values"]
                pixel_mask = batch["pixel_mask"]

                labels = [
                    {k: v for k, v in t.items()} for t in batch["labels"]
                ]  # these are in DETR format, resized + normalized

                # forward pass
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                results = im_processor.post_process_object_detection(outputs, 0, orig_target_sizes)  # convert outputs of model to COCO api

                module.add(prediction=results, reference=labels)
                del batch

        results = module.compute()
        print(results)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "object-detection",
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        runs = int(sys.argv[2])
        for i in range(runs):
            main(random_seed=True)
    else:
        main()
