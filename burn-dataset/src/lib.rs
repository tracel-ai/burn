#![warn(missing_docs)]

//! # Burn Dataset
//!
//! Burn Dataset is a library for creating and loading datasets.

#[macro_use]
extern crate derive_new;

extern crate dirs;

/// Sources for datasets.
pub mod source;

/// Transformations to be used with datasets.
pub mod transform;

/// Audio datasets.
#[cfg(feature = "audio")]
pub mod audio;

mod dataset;
pub use dataset::*;
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
