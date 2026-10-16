#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Dataset
//!
//! Datasets, transformations and data sources for Burn.
//!
//! Applications enable the `dataset` feature of `burn` (also enabled by `train`) and use this crate as
//! `burn::data::dataset`.
//!
//! - [`Dataset`] is random access by index. [`InMemDataset`] holds items in memory, and the
//!   `sqlite` feature adds `SqliteDataset` for datasets larger than memory.
//! - [`transform`]: mapping, sampling, shuffling, windowing, partial views, selection and
//!   composition of datasets, applied lazily.
//! - [`source`]: dataset downloads, including Hugging Face datasets.
//! - `vision`, `nlp`, `audio`: ready-made datasets such as MNIST, CIFAR, image folders, AG News
//!   and Speech Commands, behind the features of the same names.
//!
//! # Feature flags
//!
//! - `sqlite`: SQLite-backed datasets, using the [Turso](https://turso.tech/) engine.
//!   `sqlite-bundled` is a deprecated alias.
//! - `vision`, `nlp`, `audio`: domain datasets and loaders. `builtin-sources` enables the
//!   downloadable vision and NLP datasets.
//! - `dataframe`: datasets backed by a Polars `DataFrame`.
//! - `network`: file downloads.
//! - `fake`: generated datasets for tests.
//! - `tracing`: instrument operations with the `tracing` crate.

#[macro_use]
extern crate derive_new;

extern crate alloc;
extern crate dirs;

/// Sources for datasets.
pub mod source;

pub mod transform;

/// Audio datasets.
#[cfg(feature = "audio")]
pub mod audio;

/// Vision datasets.
#[cfg(feature = "vision")]
pub mod vision;

/// Natural language processing datasets.
#[cfg(feature = "nlp")]
pub mod nlp;

/// Network dataset utilities.
#[cfg(feature = "network")]
pub mod network {
    pub use burn_std::network::*;
}

mod dataset;
pub use dataset::*;
#[cfg(feature = "sqlite")]
pub use source::huggingface::downloader::*;

#[cfg(test)]
mod test_data {
    pub fn string_items() -> Vec<String> {
        vec![
            "1 Item".to_string(),
            "2 Items".to_string(),
            "3 Items".to_string(),
            "4 Items".to_string(),
        ]
    }
}
