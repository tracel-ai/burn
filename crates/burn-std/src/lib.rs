#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Standard Library
//!
//! Core types and utilities shared across the Burn crates.
//!
//! - [`Shape`], slicing ([`s!`]) and indexing helpers.
//! - [`DType`], the element traits and conversions, and [`TensorData`], the backend-independent
//!   representation of tensor contents.
//! - [`Distribution`] for random tensor initialization.
//! - [`DeviceSettings`], per-device defaults such as dtypes.
//! - Quantization schemes, [`Bytes`], identifiers and errors.
//! - [`config`]: runtime configuration read from `burn.toml`.
//! - `network`: file downloads with a progress bar (`network` feature).
//!
//! Applications use these through `burn::tensor`. The crate supports `no_std` with `alloc`.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `network`: file downloads.
//! - `tracing`: instrument operations with the `tracing` crate.

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

/// Runtime kind of the host program (async / sync / no-std).
pub mod runtime_kind;
pub use runtime_kind::*;

/// Distributed configurations.
pub mod distributed;

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

pub use cubecl_common::device_handle::DeviceHandle;
pub use cubecl_common::*;
pub use cubecl_environment::bytes::*;

// Environment shims live in `cubecl-environment`. They are re-exported here so
// backends and other burn crates don't have to depend on it directly.
pub use cubecl_environment::future::reader;
pub use cubecl_environment::{backtrace, future, rand, stream, sync};

/// Compute environment values, kept by the cubecl runtimes and carried by every backend.
pub mod environment {
    pub use cubecl_environment::environment::{
        EnvironmentBundle, EnvironmentError, EnvironmentId, EnvironmentOrigin,
    };
}

pub use half::{bf16, f16};

pub use cubecl_common::flex32;
