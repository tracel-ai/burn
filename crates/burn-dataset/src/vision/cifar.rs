//! CIFAR Dataset Module
//!
//! This module provides functionality for loading the CIFAR-10 and CIFAR-100 image classification datasets.
//! CIFAR (Canadian Institute For Advanced Research) datasets are widely used benchmarks in computer vision,
//! consisting of 32×32 pixel color images split into training (50,000 images) and test (10,000 images) sets.
//!
//! ## Dataset Variants
//! - **CIFAR-10**: Contains 10 distinct classes (e.g., airplane, automobile, bird, cat)
//!     - CIFAR-10 mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44).
//!     - Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
//! - **CIFAR-100**: Contains 100 fine-grained classes (e.g., beaver, dolphin, oak tree)
//!     - CIFAR-100 mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L75).
//!     - Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
//!
//! ## Usage Example
//! ```rust
//! use burn_dataset::vision::CifarDataset;
//! use burn_dataset::vision::CifarType;
//!
//! // Create a CIFAR-10 dataset accessor
//! let dataset = CifarDataset::new(CifarType::Cifar10);
//!
//! // Access training and test sets
//! let train_dataset = dataset.train();
//! let test_dataset = dataset.test();
//! ```
//! ```rust
//! use burn_dataset::vision::CifarDataset;
//! use burn_dataset::vision::CifarType;
//!
//! // Create a CIFAR-100 dataset accessor
//! let dataset = CifarDataset::new(CifarType::Cifar100);
//!
//! // Access training and test sets
//! let train_dataset = dataset.train();
//! let test_dataset = dataset.test();
//! ```

use std::{path::PathBuf, sync::Mutex};

use burn_common::network::downloader;
use flate2::read::GzDecoder;
use tar::Archive;

use crate::vision::ImageFolderDataset;

/// CIFAR-10 mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44).
/// Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
const CIFAR10_URL: &str = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz";

/// CIFAR-100 mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L75).
/// Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
const CIFAR100_URL: &str = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz";

/// Enum representing the types of CIFAR datasets available.
///
/// CIFAR (Canadian Institute For Advanced Research) datasets are widely used benchmarks for image classification.
/// This enum provides support for the two main CIFAR datasets.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum CifarType {
    /// CIFAR-10 dataset containing 10 classes with 60,000 images in total.
    Cifar10,
    /// CIFAR-100 dataset containing 100 classes with 60,000 images in total.
    Cifar100,
}

/// CIFAR dataset accessor.
///
/// This struct provides convenient access to the CIFAR-10 and CIFAR-100 image classification datasets.
/// It automatically downloads (if not already downloaded), extracts, and loads the datasets.
///
/// All images in CIFAR datasets are 32×32 pixel color images, with 50,000 images in the training set
/// and 10,000 images in the test set.
///
/// ## Differences between datasets
/// - **CIFAR-10**: Contains 10 mutually exclusive classes such as airplane, automobile, bird, cat, etc.
/// - **CIFAR-100**: Contains 100 fine-grained classes such as beaver, dolphin, etc.
pub struct CifarDataset {
    cifar_dir: PathBuf,
}

impl CifarDataset {
    /// Creates a new CIFAR dataset accessor.
    ///
    /// # Arguments
    /// * `cifar_type` - Specifies whether to use CIFAR-10 or CIFAR-100 dataset
    pub fn new(cifar_type: CifarType) -> Self {
        Self {
            cifar_dir: download(&cifar_type),
        }
    }

    /// Gets the training dataset.
    ///
    /// # Returns
    /// An `ImageFolderDataset` instance containing 50,000 training images
    pub fn train(&self) -> ImageFolderDataset {
        ImageFolderDataset::new_classification(self.cifar_dir.join("train")).unwrap()
    }

    /// Gets the test dataset.
    ///
    /// # Returns
    /// An `ImageFolderDataset` instance containing 10,000 test images
    pub fn test(&self) -> ImageFolderDataset {
        ImageFolderDataset::new_classification(self.cifar_dir.join("test")).unwrap()
    }
}

/// CIFAR dataset download lock.
///
/// This lock ensures that only one thread downloads the CIFAR dataset at a time.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

fn download(cifar_type: &CifarType) -> PathBuf {
    // Acquire the lock. This will block if another thread already holds the lock.
    let _lock = DOWNLOAD_LOCK.lock().unwrap();

    // Dataset files are stored in the burn-dataset cache directory
    let cache_dir = dirs::home_dir()
        .expect("Could not get home directory")
        .join(".cache")
        .join("burn-dataset");

    // Cifar store directory
    let cifar_dir = match cifar_type {
        CifarType::Cifar10 => cache_dir.join("cifar10"),
        CifarType::Cifar100 => cache_dir.join("cifar100"),
    };

    // Cifar dataset url
    let url = match cifar_type {
        CifarType::Cifar10 => CIFAR10_URL,
        CifarType::Cifar100 => CIFAR100_URL,
    };

    // Cifar dataset archive filename
    let filename = match cifar_type {
        CifarType::Cifar10 => "cifar10.tgz",
        CifarType::Cifar100 => "cifar100.tgz",
    };

    // Check for already downloaded content
    if !cifar_dir.exists() {
        // Download gzip file
        let bytes = downloader::download_file_as_bytes(url, filename);

        // Decode gzip file content and unpack archive
        let gz_buffer = GzDecoder::new(&bytes[..]);
        let mut archive = Archive::new(gz_buffer);
        archive.unpack(cache_dir).unwrap();
    }

    cifar_dir
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dataset, vision::Annotation};

    /// CIFAR dataset length
    const TRAINDATASET_LEN: usize = 50000;
    const TESTDATASET_LEN: usize = 10000;

    /// CIFAR-10 label range
    const CIFAR10_LABEL_MIN: usize = 0;
    const CIFAR10_LABEL_MAX: usize = 9;

    /// CIFAR-100 label range
    const CIFAR100_LABEL_MIN: usize = 0;
    const CIFAR100_LABEL_MAX: usize = 99;

    #[test]
    fn test_cifar10_download() {
        let cifar_dir = download(&CifarType::Cifar10);
        assert!(cifar_dir.exists());
    }

    #[test]
    fn test_cifar100_download() {
        let cifar_dir = download(&CifarType::Cifar100);
        assert!(cifar_dir.exists());
    }

    #[test]
    fn test_cifar10_len() {
        let dataset = CifarDataset::new(CifarType::Cifar10);
        let train_dataset = dataset.train();
        let test_dataset = dataset.test();
        assert_eq!(train_dataset.len(), TRAINDATASET_LEN);
        assert_eq!(test_dataset.len(), TESTDATASET_LEN);
    }

    #[test]
    fn test_cifar100_len() {
        let dataset = CifarDataset::new(CifarType::Cifar100);
        let train_dataset = dataset.train();
        let test_dataset = dataset.test();
        assert_eq!(train_dataset.len(), TRAINDATASET_LEN);
        assert_eq!(test_dataset.len(), TESTDATASET_LEN);
    }

    #[test]
    fn test_cifar10_label_range() {
        let dataset = CifarDataset::new(CifarType::Cifar10);
        let test_dataset = dataset.test();
        let (min, max) = get_label_range(&test_dataset);
        assert_eq!(min, CIFAR10_LABEL_MIN);
        assert_eq!(max, CIFAR10_LABEL_MAX);
    }

    #[test]
    fn test_cifar100_label_range() {
        let dataset = CifarDataset::new(CifarType::Cifar100);
        let test_dataset = dataset.test();
        let (min, max) = get_label_range(&test_dataset);
        assert_eq!(min, CIFAR100_LABEL_MIN);
        assert_eq!(max, CIFAR100_LABEL_MAX);
    }

    fn get_label_range(dataset: &ImageFolderDataset) -> (usize, usize) {
        let labels: Vec<_> = dataset.iter().map(|item| item.annotation).collect();
        let mut min = 128;
        let mut max = 0;
        for label in labels {
            let index = match label {
                Annotation::Label(index) => index,
                _ => 0,
            };
            if index < min {
                min = index;
            }
            if index > max {
                max = index;
            }
        }

        (min, max)
    }
}
