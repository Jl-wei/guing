import os
import pandas as pd
import datasets

_VERSION = datasets.Version("0.0.8")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_CATEGORIES = ["Screen"]
_DEFAULT_CONFIG = datasets.BuilderConfig(name="google-detect", version=_VERSION)

class GoogleDetect(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "google-detect"

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int32"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("string"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/detection/data.jsonl")
        images_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/detection/png_images")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir):
        def process_annot(span):
            return {
                "id": span["id"],
                "area": span["width"]*span["height"],
                "bbox": [span["x"], span["y"], span["width"], span["height"]],
                "category": span["label"],
            }
        
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            image_path = os.path.join(images_dir, row["image"])
            image = open(image_path, "rb").read()
            objects = [process_annot(span) for span in row["spans"]]
            
            yield row["image"], {
                "image_id": _,
                "image": {"path": image_path, "bytes": image},
                "width": row["width"],
                "height": row["height"],
                "objects": objects,
            }
