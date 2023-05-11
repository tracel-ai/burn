import abc
import json
import argparse
import numpy as np

from datasets import load_dataset
from typing import List, Any, Tuple
from tqdm import tqdm

from json import JSONEncoder


class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


DOWNLOAD_DIR = ".cache/burn-dataset"


class Extractor(abc.ABC):
    def extract(self, item: Any) -> Any:
        pass

    @abc.abstractproperty
    def name(self) -> str:
        pass


class RawFieldExtractor(Extractor):
    def __init__(self, field_name: str):
        self.field_name = field_name

    def extract(self, item: Any) -> Any:
        return item[self.field_name]

    @property
    def name(self) -> str:
        return self.field_name


class ImageFieldExtractor(Extractor):
    def __init__(self, field_name: str):
        self.field_name = field_name

    def extract(self, item: Any) -> Any:
        image = item[self.field_name]
        return np.array(image).tolist()

    @property
    def name(self) -> str:
        return self.field_name


def download(
    name: str,
    keys: List[str],
    download_file: str,
    extractors: List[Extractor],
    *config,
    **kwargs,
):
    dataset_all = load_dataset(name, *config, **kwargs)
    for key in keys:
        dataset = dataset_all[key]
        dataset_file = f"{download_file}-{key}"
        print(f"Saving dataset: {name} - {key}")

        with open(dataset_file, "w") as file:
            for item in tqdm(dataset):
                payload = {}
                for extactor in extractors:
                    payload[extactor.name] = extactor.extract(item)

                payload = json.dumps(payload, cls=CustomEncoder)
                line = f"{payload}\n"
                file.write(line)


def config_named(value: str) -> Tuple[str, str]:
    try:
        key, value = value.split("=")
        return {key: value}
    except:
        raise argparse.ArgumentTypeError("config_named must be key=value")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huggingface datasets downloader to use with burn-dataset"
    )
    parser.add_argument(
        "--name", type=str, help="Name of the dataset to download", required=True
    )
    parser.add_argument(
        "--file", type=str, help="Base file name where the data is saved", required=True
    )
    parser.add_argument(
        "--split", type=str, help="Splits to downloads", nargs="+", required=True
    )
    parser.add_argument(
        "--config", type=str, help="Config of the dataset", nargs="+", default=[]
    )
    parser.add_argument(
        "--config-named",
        type=config_named,
        help="Named config of the dataset",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--extract-image",
        type=str,
        help="Image field to extract",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--extract-raw", type=str, help="Raw field to extract", nargs="+", default=[]
    )

    return parser.parse_args()


def run():
    args = parse_args()
    extractors = []

    for field_name in args.extract_image:
        extractors.append(ImageFieldExtractor(field_name))

    for field_name in args.extract_raw:
        extractors.append(RawFieldExtractor(field_name))

    kwargs = {}
    for config_named in args.config_named:
        kwargs = kwargs | config_named

    download(
        args.name,
        args.split,
        args.file,
        extractors,
        *args.config,
        **kwargs,
    )


if __name__ == "__main__":
    run()
