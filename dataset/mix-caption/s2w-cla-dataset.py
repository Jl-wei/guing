import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.1")

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

_DEFAULT_CONFIG = datasets.BuilderConfig(name="cap-mix-s2w-cla", version=_VERSION)


class MixCapion(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "cap-mix-s2w-cla"

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
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/mix-caption/s2w-cla.jsonl")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                },
            ),
        ]

    def _generate_examples(self, metadata_path):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            image_path = row["image_name"]
            image = open(image_path, "rb").read()

            yield row["id"], {
                "caption": row["caption"],
                "image": image_path,
            }
