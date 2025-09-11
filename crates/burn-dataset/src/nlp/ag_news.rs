//! AG NEWS Dataset Module
//!
//! This module provides functionality for loading the AG NEWS text classification dataset.
//! AG NEWS is a collection of news articles categorized into different topics.
//! The dataset is split into training (120,000 articles) and test (7,600 articles) sets.
//!
//! ## Dataset Details
//! - **Classes**: 4 categories (World, Sports, Business, Sci/Tech)
//! - **AG NEWS mirror**: [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L83)
//! - **License**: [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE)
//!
//! ## Usage Example
//! ```rust
//! use burn_dataset::nlp::AgNewsDataset;
//!
//! // Create an AG NEWS dataset accessor
//! let dataset = AgNewsDataset::new();
//!
//! // Access training and test sets
//! let train_dataset = dataset.train();
//! let test_dataset = dataset.test();
//! ```

use std::{path::PathBuf, sync::Mutex};

use burn_common::network::downloader;
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use tar::Archive;

use crate::InMemDataset;

/// AG NEWS mirror from [fastai](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L83).
/// Licensed under the [Apache License](https://github.com/fastai/fastai/blob/master/LICENSE).
const AG_NEWS_URL: &str = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz";

/// Represents an item in the AG NEWS dataset.
///
/// Each item contains a label, title, and content of a news article.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AgNewsItem {
    /// The category label of the news article.
    pub label: String,
    /// The title of the news article.
    pub title: String,
    /// The content/body of the news article.
    pub content: String,
}

/// AG NEWS dataset accessor.
///
/// This struct provides convenient access to the AG NEWS text classification dataset.
/// It automatically downloads (if not already downloaded), extracts, and loads the datasets.
///
/// The dataset is split into training (120,000 articles) and test (7,600 articles) sets.
pub struct AgNewsDataset {
    agnews_dir: PathBuf,
}

/// AG NEWS dataset download lock.
///
/// This lock ensures that only one thread downloads the AG NEWS dataset at a time.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

impl AgNewsDataset {
    /// Creates a new AG NEWS dataset accessor.
    ///
    /// This will download and extract the dataset if it's not already present.
    pub fn new() -> Self {
        Self {
            agnews_dir: Self::download(),
        }
    }

    /// Downloads and extracts the AG NEWS dataset.
    ///
    /// # Returns
    /// Path to the directory containing the extracted dataset.
    fn download() -> PathBuf {
        // Acquire the lock. This will block if another thread already holds the lock.
        let _lock = DOWNLOAD_LOCK.lock().unwrap();

        // Dataset files are stored in the burn-dataset cache directory
        let cache_dir = dirs::home_dir()
            .expect("Could not get home directory")
            .join(".cache")
            .join("burn-dataset");

        // AG NEWS dataset directory
        let agnews_dir = cache_dir.join("ag_news_csv");

        // AG NEWS dataset url
        let url = AG_NEWS_URL;

        // AG NEWS dataset archive filename
        let filename = "ag_news_csv.tgz";

        // Check for already downloaded content
        if !agnews_dir.exists() {
            // Download gzip file
            let bytes = downloader::download_file_as_bytes(url, filename);

            // Decode gzip file content and unpack archive
            let gz_buffer = GzDecoder::new(&bytes[..]);
            let mut archive = Archive::new(gz_buffer);
            archive.unpack(cache_dir).unwrap();
        }

        agnews_dir
    }

    /// Parses a CSV file into an in-memory dataset.
    ///
    /// # Arguments
    /// * `file_path` - Path to the CSV file to parse.
    ///
    /// # Returns
    /// An `InMemDataset` containing the parsed data.
    fn parse_csv(file_path: &str) -> InMemDataset<AgNewsItem> {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.has_headers(false);

        InMemDataset::from_csv(file_path, &rdr).expect("Failed to parse CSV file")
    }

    /// Gets the training dataset.
    ///
    /// # Returns
    /// An `InMemDataset` instance containing 120,000 training articles.
    pub fn train(&self) -> InMemDataset<AgNewsItem> {
        let file_path = self.agnews_dir.join("train.csv");
        Self::parse_csv(file_path.to_str().unwrap())
    }

    /// Gets the test dataset.
    ///
    /// # Returns
    /// An `InMemDataset` instance containing 7,600 test articles.
    pub fn test(&self) -> InMemDataset<AgNewsItem> {
        let file_path = self.agnews_dir.join("test.csv");
        Self::parse_csv(file_path.to_str().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    // AG NEWS dataset train and test dataset lengths
    const TRAIN_DATASET_LEN: usize = 120000;
    const TEST_DATASET_LEN: usize = 7600;

    #[test]
    fn test_agnews_download() {
        let agnews_dir = AgNewsDataset::download();
        assert!(agnews_dir.exists());
    }

    #[test]
    fn test_agnews_len() {
        let agnews = AgNewsDataset::new();
        let train_dataset = agnews.train();
        let test_dataset = agnews.test();
        assert_eq!(train_dataset.len(), TRAIN_DATASET_LEN);
        assert_eq!(test_dataset.len(), TEST_DATASET_LEN);
    }

    #[test]
    fn test_agnews_first_and_last_item() {
        let agnews = AgNewsDataset::new();

        // Test the first and the last item in training dataset
        let train_dataset = agnews.train();
        let first_item = train_dataset.get(0).unwrap();
        let last_item = train_dataset.get(train_dataset.len() - 1).unwrap();
        assert!(compare_item(&first_item, &("3".to_string(), "Wall St. Bears Claw Back Into the Black (Reuters)".to_string(), "Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.".to_string())));
        assert!(compare_item(
            &last_item,
            &(
                "2".to_string(),
                "Nets get Carter from Raptors".to_string(),
                "INDIANAPOLIS -- All-Star Vince Carter was traded by the Toronto Raptors to the New Jersey Nets for Alonzo Mourning, Eric Williams, Aaron Williams, and a pair of first-round draft picks yesterday.".to_string()
            )
        ));

        // Test the first and the last item in test dataset
        let test_dataset = agnews.test();
        let first_item = test_dataset.get(0).unwrap();
        let last_item = test_dataset.get(test_dataset.len() - 1).unwrap();
        assert!(compare_item(
            &first_item,
            &(
                "3".to_string(),
                "Fears for T N pension after talks".to_string(),
                "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.".to_string()
            )
        ));
        assert!(compare_item(
            &last_item,
            &(
                "3".to_string(),
                "EBay gets into rentals".to_string(),
                "EBay plans to buy the apartment and home rental service Rent.com for \\$415 million, adding to its already exhaustive breadth of offerings.".to_string()
            )
        ));
    }

    fn compare_item(item: &AgNewsItem, target: &(String, String, String)) -> bool {
        item.label == target.0 && item.title == target.1 && item.content == target.2
    }
}
