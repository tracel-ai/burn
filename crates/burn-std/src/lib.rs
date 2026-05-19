#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Standard Library
//!
//! This library contains core types and utilities shared across Burn, including shapes, indexing,
//! and data types.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Id module contains types for unique identifiers.
pub mod id;

/// Tensor utilities.
pub mod tensor;
pub use tensor::*;

/// Tensor data representation and helpers.
pub mod data;
pub use data::*;

/// Random value distributions.
pub mod distribution;
pub use distribution::*;

/// Traits for tensor element types and conversions.
pub mod element;
pub use element::*;

mod device_settings;
pub use device_settings::*;

/// Configuration types for tensor operations (conv, pool, interpolate, pad, etc).
pub mod ops;
pub use ops::*;

/// Burn runtime configurations.
pub mod config;

/// Common Errors.
pub use cubecl_zspace::errors::{self, *};

/// Network utilities.
#[cfg(feature = "network")]
pub mod network;

pub struct CommunicationId;

pub use cubecl_common::bytes::*;
pub use cubecl_common::device_handle::DeviceHandle;
pub use cubecl_common::*;
pub use half::{bf16, f16};

pub use cubecl_common::flex32;
