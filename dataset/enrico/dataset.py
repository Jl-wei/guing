import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.4")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Value("string"),
        "topic": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="enrico", version=_VERSION)


class Enrico(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "enrico"

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
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/data.csv")
        images_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/enrico/screenshots")

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
        metadata = pd.read_csv(metadata_path)

        for _, row in metadata.iterrows():
            image_path = os.path.join(images_dir, f"{row['screen_id']}.jpg")
            image = open(image_path, "rb").read()

            yield row["screen_id"], {
                "image": image_path,
                "topic": row["topic"],
            }
