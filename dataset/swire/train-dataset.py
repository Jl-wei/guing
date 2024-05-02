import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "sketch": datasets.Value("string"),
        "screenshot": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="train-swire", version=_VERSION)


class TrainSwire(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "train-swire"

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
        metadata_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/train-data.jsonl")
        sketches_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/sketches")
        screenshots_dir = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "dataset/swire/screenshots")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "sketches_dir": sketches_dir,
                    "screenshots_dir": screenshots_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, sketches_dir, screenshots_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            sketch_name = row["sketch"]
            screenshot_name = row["screenshot"]
            
            sketch_path = os.path.join(sketches_dir, sketch_name)
            screenshot_path = os.path.join(screenshots_dir, screenshot_name)
            
            yield row["sketch"], {
                "sketch": sketch_path,
                "screenshot": screenshot_path,
            }