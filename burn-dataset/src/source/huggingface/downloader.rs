use dirs::home_dir;
use std::fs;
use std::process::Command;

pub enum Extractor {
    Raw(String),
    Image(String),
}

pub fn download(
    name: String,
    splits: Vec<String>,
    base_file: String,
    extractors: Vec<Extractor>,
    config: Vec<String>,
    config_named: Vec<(String, String)>,
) {
    let mut command = Command::new("python");

    command.arg(dataset_downloader_file_path());

    command.arg("--file");
    command.arg(base_file);

    command.arg("--name");
    command.arg(name);

    command.arg("--split");
    for split in splits {
        command.arg(split);
    }

    for extractor in extractors {
        match extractor {
            Extractor::Raw(field) => {
                command.arg("--extract-raw");
                command.arg(field);
            }
            Extractor::Image(field) => {
                command.arg("--extract-image");
                command.arg(field);
            }
        };
    }

    if config.len() > 0 {
        command.arg("--config");
        for config in config {
            command.arg(config);
        }
    }
    if config_named.len() > 0 {
        command.arg("--config-named");
        for (key, value) in config_named {
            command.arg(format!("{}={}", key, value));
        }
    }

    println!("{:?}", command);
    let output = command.output().unwrap();
    println!("{:?}", output);
}

pub(crate) fn cache_dir() -> String {
    let home_dir = home_dir().unwrap();
    let home_dir = home_dir.to_str().map(|s| s.to_string());
    let home_dir = home_dir.unwrap();
    format!("{}/.cache/burn-dataset", home_dir)
}

fn dataset_downloader_file_path() -> String {
    let path_dir = cache_dir();
    let path_file = format!("{}/dataset.py", path_dir);

    fs::write(path_file.as_str(), PYTHON_SOURCE).expect("Write python dataset downloader");
    path_file
}

const PYTHON_SOURCE: &str = r#"
import os
import abc
import json
import argparse
import numpy as np

from datasets import load_dataset
from typing import List, Any, Tuple
from tqdm import tqdm


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
    download_dir: str,
    download_file: str,
    extractors: List[Extractor],
    *config,
    **kwargs,
):
    dataset_all = load_dataset(name, *config, **kwargs)
    for key in keys:
        dataset = dataset_all[key]
        dataset_file = os.path.join(download_dir, f"{download_file}-{key}")
        print(f"Saving dataset: {name} - {key}")

        with open(dataset_file, "w") as file:
            for item in tqdm(dataset):
                payload = {}
                for extactor in extractors:
                    payload[extactor.name] = extactor.extract(item)

                payload = json.dumps(payload)
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

    home = os.path.expanduser("~")
    download_dir = str(os.path.join(home, DOWNLOAD_DIR))
    os.makedirs(download_dir, exist_ok=True)

    kwargs = {}
    for config_named in args.config_named:
        kwargs = kwargs | config_named

    download(
        args.name,
        args.split,
        download_dir,
        args.file,
        extractors,
        *args.config,
        **kwargs,
    )


if __name__ == "__main__":
    run()
"#;
