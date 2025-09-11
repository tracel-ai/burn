//! FDU CLF Dataset Module
//!
//! This module provides functionality for loading the Fudan University Chinese Text Classification Dataset.
//!
//! ## Dataset Information
//! - **Source**: [Fudan University Chinese Text Classification Dataset](https://gitcode.com/open-source-toolkit/6a679)
//! - **License**: [MIT License](https://gitcode.com/open-source-toolkit/6a679/blob/main/LICENSE)
//! - **Content**: Chinese text documents categorized into different classes
//!
//! ## Usage Example
//! ```rust
//! use burn_dataset::nlp::FduClfDataset;
//!
//! // Create a FDU CLF dataset accessor
//! let dataset = FduClfDataset::new();
//!
//! // Access training and test sets
//! let train_dataset = dataset.train();
//! let test_dataset = dataset.test();
//! ```

use crate::nlp::TextFolderDataset;
use burn_common::network::downloader;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;
use zip::ZipArchive;

/// FDU CLF dataset accessor.
///
/// This struct provides convenient access to the Fudan University Chinese Text Classification Dataset.
/// It automatically downloads (if not already downloaded), extracts, and loads the dataset.
///
/// The dataset consists of Chinese text documents categorized into different classes,
/// with 9,804 documents in the training set and 9,832 documents in the test set.
pub struct FduClfDataset {
    fdu_clf_dir: PathBuf,
}

/// FDU-CLF dataset download lock.
///
/// This lock ensures that only one thread downloads the FDU-CLF dataset at a time.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

impl FduClfDataset {
    /// Creates a new FDU CLF dataset accessor.
    ///
    /// This function automatically downloads (if not already downloaded), extracts, and loads the dataset.
    ///
    /// # Returns
    /// A `FduClfDataset` instance
    pub fn new() -> Self {
        let fdu_clf_dir = Self::download().unwrap();
        Self { fdu_clf_dir }
    }

    /// Gets the training dataset.
    ///
    /// # Returns
    /// A `TextFolderDataset` instance containing 9,804 training documents
    pub fn train(&self) -> TextFolderDataset {
        TextFolderDataset::new_classification(&self.fdu_clf_dir.join("train")).unwrap()
    }

    /// Gets the test dataset.
    ///
    /// # Returns
    /// A `TextFolderDataset` instance containing 9,832 test documents
    pub fn test(&self) -> TextFolderDataset {
        TextFolderDataset::new_classification(&self.fdu_clf_dir.join("test")).unwrap()
    }

    /// Downloads and extracts the FDU CLF dataset.
    ///
    /// This function handles the download, extraction, and preparation of the Fudan University Chinese Text Classification Dataset.
    /// It follows these steps:
    /// 1. Checks if the dataset directory already exists
    /// 2. If not, downloads the dataset archive
    /// 3. Extracts the main archive
    /// 4. Renames the extracted directory
    /// 5. Extracts the train and test archives
    /// 6. Removes temporary archive files
    ///
    /// # Returns
    /// A `Result` containing the path to the dataset directory if successful, or an error if something went wrong
    fn download() -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Acquire the lock. This will block if another thread already holds the lock.
        let _lock = DOWNLOAD_LOCK.lock().unwrap();

        // Dataset files are stored in the burn-dataset cache directory
        let cache_dir = dirs::home_dir()
            .expect("Could not get home directory")
            .join(".cache")
            .join("burn-dataset");
        let fdu_clf_dir = cache_dir.join("fdu_clf");

        // If the dataset directory already exists, return it
        if fdu_clf_dir.exists() {
            return Ok(fdu_clf_dir);
        }

        // Download fudan.zip file
        let bytes = downloader::download_file_as_bytes(
            "https://raw-cdn.gitcode.com/open-source-toolkit/6a679/blobs/41346c70dbc2dae5c5b1824e200a2cd4639fdefd/fudan.zip",
            "fdu_clf.zip",
        );

        // Unzip fudan.zip
        let cursor = Cursor::new(bytes);
        let mut archive = ZipArchive::new(cursor)?;
        archive.extract(&cache_dir)?;
        // Rename fudan to fdu_clf
        std::fs::rename(&cache_dir.join("fudan"), &fdu_clf_dir)?;

        // Unzip test.zip
        let test_zip_path = fdu_clf_dir.join("test.zip");
        let test_zip_bytes = std::fs::read(&test_zip_path)?;
        let test_cursor = Cursor::new(test_zip_bytes);
        let mut test_archive = ZipArchive::new(test_cursor)?;
        std::fs::create_dir_all(&fdu_clf_dir)?;
        test_archive.extract(&fdu_clf_dir)?;
        // Remove test.zip
        std::fs::remove_file(&test_zip_path)?;

        // Unzip train.zip
        let train_zip_path = &fdu_clf_dir.join("train.zip");
        let train_zip_bytes = std::fs::read(&train_zip_path)?;
        let train_cursor = Cursor::new(train_zip_bytes);
        let mut train_archive = ZipArchive::new(train_cursor)?;
        std::fs::create_dir_all(&fdu_clf_dir)?;
        train_archive.extract(&fdu_clf_dir)?;
        // Remove train.zip
        std::fs::remove_file(&train_zip_path)?;

        Ok(fdu_clf_dir)
    }
}
