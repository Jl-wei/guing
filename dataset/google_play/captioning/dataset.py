import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.5")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Value("string"),
        "caption": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="google-caption", version=_VERSION)


class GoogleCaption(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "google-caption"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/captioning/data.jsonl")
        images_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/google_play/captioning/png_images")

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
                "caption": row["caption"],
                "image": image_path,
            }