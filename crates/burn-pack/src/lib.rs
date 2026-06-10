#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Pack
//!
//! The **burnpack** binary serialization format for the Burn deep learning framework.
//!
//! `burn-pack` is intentionally minimal and tensor-library-agnostic: it depends only on
//! `burn-std` (for [`DType`](burn_std::DType) / [`Bytes`](burn_std::Bytes)), `serde`, and a
//! CBOR codec. It knows how to read and write the burnpack container format but has no notion
//! of Burn modules or tensors. Higher layers (e.g. `burn-core`) bridge between [`Tensor`]
//! entries and their own tensor/snapshot types.
//!
//! ## File format
//!
//! ```text
//! ┌──────────────────────────────────┐
//! │  Header (10 bytes)               │
//! │  - Magic number (4 bytes)        │  0x4255524E ("BURN")
//! │  - Version (2 bytes)             │  Format version
//! │  - Metadata size (4 bytes)       │  Size of CBOR metadata (u32, little-endian)
//! ├──────────────────────────────────┤
//! │  Metadata (CBOR)                 │  [`Metadata`]: per-tensor [`TensorDescriptor`]
//! │                                  │  (dtype, shape, data offsets, optional param id)
//! │                                  │  plus user key/value metadata.
//! ├──────────────────────────────────┤
//! │  Tensor data section             │  Each tensor aligned to 256 bytes
//! │                                  │  ([`TENSOR_ALIGNMENT`]) for mmap zero-copy loading.
//! └──────────────────────────────────┘
//! ```
//!
//! Read a pack with [`Reader`], write one with [`Writer`]; both operate on [`Tensor`] entries
//! that carry the format-level metadata plus a lazy provider of the raw little-endian bytes.
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and memory mapping (default)
//! - `memmap`: Enables memory-mapped zero-copy file loading (default, implies `std`)

extern crate alloc;

mod base;
mod reader;
mod tensor;
mod writer;

pub use base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAGIC_NUMBER, Metadata, TENSOR_ALIGNMENT,
    TensorDescriptor,
};
pub use reader::Reader;
pub use tensor::{Tensor, TensorBytesFn};
pub use writer::Writer;
