#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # Burn Common Library
//!
//! This library contains common types used by other Burn crates that must be shared.

/// Id module contains types for unique identifiers.
pub mod id;

pub use cubecl_common::*;

extern crate alloc;

/// Network utilities.
#[cfg(feature = "network")]
pub mod network;

/// Parallel utilities.
pub mod parallel;
