use flate2::read::GzDecoder;
use std::path::{Path, PathBuf};
use tar::Archive;

use burn::data::{dataset::vision::ImageFolderDataset, network::downloader};

/// CIFAR-10 mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44).
/// Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
const URL: &str = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz";

/// The [CIFAR-10](https://www.cs.toronto.edu/%7Ekriz/cifar.html) dataset consists of 60,000 32x32
/// colour images, with 6,000 images per class. There are 50,000 training images and 10,000 test
/// images.
///
/// The data is downloaded from the web from the [fastai mirror](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44).
pub trait CIFAR10Loader {
    fn cifar10_train() -> Self;
    fn cifar10_test() -> Self;
}

impl CIFAR10Loader for ImageFolderDataset {
    /// Creates a new CIFAR10 train dataset.
    fn cifar10_train() -> Self {
        let root = download();

        Self::new_classification(root.join("train")).unwrap()
    }

    /// Creates a new CIFAR10 test dataset.
    fn cifar10_test() -> Self {
        let root = download();

        Self::new_classification(root.join("test")).unwrap()
    }
}

/// Download the CIFAR10 dataset from the web to the current example directory.
fn download() -> PathBuf {
    // Point to current example directory
    let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
    let cifar_dir = example_dir.join("cifar10");

    // Check for already downloaded content
    let labels_file = cifar_dir.join("labels.txt");
    if !labels_file.exists() {
        // Download gzip file
        let bytes = downloader::download_file_as_bytes(URL, "cifar10.tgz");

        // Decode gzip file content and unpack archive
        let gz_buffer = GzDecoder::new(&bytes[..]);
        let mut archive = Archive::new(gz_buffer);
        archive.unpack(example_dir).unwrap();
    }

    cifar_dir
}
