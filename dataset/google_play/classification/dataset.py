import pandas as pd
import datasets
import os
from datasets.tasks import ImageClassification

_VERSION = datasets.Version("0.0.4")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Value("string"),
        "labels": datasets.features.ClassLabel(names=[0,1,2]),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="google-classify", version=_VERSION)


class GoogleClassify(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "google-classify"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="labels")],
        )

    def _split_generators(self, dl_manager):
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/classification/data.jsonl")
        images_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/classification/png_images")

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
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            image_path = os.path.join(images_dir, row["image"])

            yield row["image"], {
                "labels": row['label'],
                "image": image_path,
            }
