#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Dataset
//!
//! Burn Dataset is a library for creating and loading datasets.

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
pub mod network;

mod dataset;
pub use dataset::*;
#[cfg(any(feature = "sqlite", feature = "sqlite-bundled"))]
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
