#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Burn
//! This library strives to serve as a comprehensive **deep learning framework**,
//! offering exceptional flexibility and written in Rust. The main objective is to cater
//! to both researchers and practitioners by simplifying the process of experimenting,
//! training, and deploying models.

pub use burn_core::*;

/// Train module
#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}
