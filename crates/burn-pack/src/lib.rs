#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Pack
//!
//! The **burnpack** binary serialization format for the Burn deep learning framework.
//!
//! `burn-pack` is intentionally minimal and tensor-library-agnostic: it depends only on
//! `burn-std` (for [`DType`] / [`Bytes`]), `serde`, and a CBOR codec. It knows how to read
//! and write the burnpack container format but has no notion of Burn modules or tensors.
//! Higher layers (e.g. `burn-core`) bridge between [`Tensor`] entries and their own
//! tensor/snapshot types.
//!
//! Write a pack with [`Writer`], read one with [`Reader`]; both operate on [`Tensor`]
//! entries that carry the format-level metadata plus a lazy provider of the raw
//! little-endian bytes.
//!
//! ```
//! use burn_pack::{Bytes, DType, Reader, Tensor, Writer};
//!
//! // A 2x2 f32 tensor, as raw little-endian bytes.
//! let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
//!     .iter()
//!     .flat_map(|v| v.to_le_bytes())
//!     .collect();
//! let tensor = Tensor::new(
//!     "weight".to_string(),
//!     DType::F32,
//!     vec![2, 2],
//!     Some(42), // optional param id
//!     Bytes::from_bytes_vec(raw),
//! );
//!
//! // Write to an in-memory buffer ...
//! let packed = Writer::new(vec![tensor])
//!     .with_metadata("producer", "burn-pack docs")
//!     .into_bytes()
//!     .unwrap();
//!
//! // ... and read it back.
//! let reader = Reader::from_bytes(packed).unwrap();
//! assert_eq!(reader.metadata()["producer"], "burn-pack docs");
//! // Consume the reader to get the tensors (zero-copy views into the source).
//! let tensors = reader.into_tensors().unwrap();
//! assert_eq!(tensors.len(), 1);
//! assert_eq!(tensors[0].name, "weight");
//! assert_eq!(tensors[0].shape.to_vec(), vec![2, 2]);
//! assert_eq!(tensors[0].param_id, Some(42));
//! ```
//!
//! # File format
//!
//! A burnpack file has three parts: a fixed-size header, a CBOR metadata blob, and a
//! 256-byte-aligned tensor data section. All multi-byte integers are little-endian.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │ Header — 10 bytes ([`HEADER_SIZE`])                           │
//! │   magic         : u32  — 0x4255524E "BURN" ([`MAGIC_NUMBER`]) │
//! │   version       : u16  — format version ([`FORMAT_VERSION`])  │
//! │   metadata_size : u32  — byte length of the CBOR metadata     │
//! ├──────────────────────────────────────────────────────────────┤
//! │ Metadata — CBOR, `metadata_size` bytes                       │
//! │   tensors : map<name, descriptor>                            │
//! │     dtype        : [`DType`]                                  │
//! │     shape        : list<u64>                                  │
//! │     data_offsets : (start, end)  relative to the data section │
//! │     param_id     : optional u64  (training-state identity)    │
//! │   metadata : map<string, string>  user key/value pairs       │
//! ├──────────────────────────────────────────────────────────────┤
//! │ Padding to the next 256-byte boundary                        │
//! │   ([`aligned_data_section_start`])                            │
//! ├──────────────────────────────────────────────────────────────┤
//! │ Tensor data section                                          │
//! │   each tensor's bytes start on a 256-byte boundary           │
//! │   ([`TENSOR_ALIGNMENT`]) for aligned, lazy file-backed       │
//! │   loading (see [`Bytes::from_file`]).                         │
//! │   tensors sliced zero-copy.                                   │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why 256-byte alignment
//!
//! Aligning every tensor to a 256-byte boundary ([`TENSOR_ALIGNMENT`]) lets a reader
//! memory-map the file and hand out tensor slices without copying, while satisfying the
//! alignment requirements of every element type (including 8-byte `f64`), cache lines,
//! and GPU coalesced access. 256 bytes matches the choice made by GGUF, MLX, ncnn, and
//! other major formats.
//!
//! ## Safety limits
//!
//! Reading is hardened against malicious or corrupt inputs. The reader rejects files
//! that exceed any of the following before allocating for them:
//!
//! - [`MAX_METADATA_SIZE`] — largest CBOR metadata blob
//! - [`MAX_TENSOR_COUNT`] — largest number of tensors
//! - [`MAX_TENSOR_SIZE`] — largest single tensor
//! - [`MAX_CBOR_RECURSION_DEPTH`] — deepest CBOR nesting (stack-overflow guard)
//! - [`MAX_FILE_SIZE`] — largest file accepted by the file loaders (std only)
//!
//! It also validates that the file is large enough to contain every tensor it claims,
//! returning [`Error::ValidationError`] otherwise.
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O ([`Reader::from_file`] / [`Writer::write_to_file`]) (default)

extern crate alloc;

mod base;
mod reader;
mod tensor;
mod writer;

#[cfg(feature = "std")]
pub use base::MAX_FILE_SIZE;
pub use base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAGIC_NUMBER, MAX_CBOR_RECURSION_DEPTH,
    MAX_METADATA_SIZE, MAX_TENSOR_COUNT, MAX_TENSOR_SIZE, Scalar, ScalarConversionError,
    TENSOR_ALIGNMENT, aligned_data_section_start,
};
pub use reader::Reader;
pub use tensor::Tensor;
pub use writer::Writer;

/// The canonical file extension for burnpack files (without the leading dot).
pub const EXTENSION: &str = "bpk";

// Re-export the core types so callers can build [`Tensor`] entries and inspect descriptors
// without depending on `burn-std` directly.
pub use burn_std::{Bytes, DType, Shape};
