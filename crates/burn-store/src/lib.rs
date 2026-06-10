#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Store
//!
//! The **burnpack** binary serialization format for the Burn deep learning framework.
//!
//! This crate is intentionally minimal and tensor-library-agnostic: it depends only on
//! `burn-std` (for [`DType`](burn_std::DType) / [`Bytes`](burn_std::Bytes)), `serde`, and a
//! CBOR codec. It knows how to read and write the burnpack container format but has no notion
//! of Burn modules or tensors. Higher layers (e.g. `burn-core`) bridge between
//! [`BurnpackTensor`] entries and their own tensor/snapshot types.
//!
//! ## Format
//!
//! See the [`burnpack`] module for the complete file format specification (magic header,
//! CBOR metadata, 256-byte aligned tensor data section enabling mmap zero-copy loading).
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and memory mapping (default)
//! - `memmap`: Enables memory-mapped zero-copy file loading (default, implies `std`)
//! - `burnpack`: Enables the burnpack reader/writer (default)

extern crate alloc;

#[cfg(feature = "burnpack")]
pub mod burnpack;

#[cfg(feature = "burnpack")]
pub use burnpack::base::BurnpackError;
#[cfg(feature = "burnpack")]
pub use burnpack::reader::BurnpackReader;
#[cfg(feature = "burnpack")]
pub use burnpack::tensor::{BurnpackTensor, TensorBytesFn};
#[cfg(feature = "burnpack")]
pub use burnpack::writer::BurnpackWriter;
