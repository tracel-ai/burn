#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Store
//!
//! Advanced model storage and serialization infrastructure for the Burn deep learning framework.
//!
//! This crate provides comprehensive functionality for storing and loading Burn modules
//! and their tensor data, with support for cross-framework interoperability, flexible filtering,
//! and efficient memory management through lazy materialization.
//!
//! ## Key Features
//!
//! - **Burnpack Format**: Native Burn format with CBOR metadata, ParamId persistence for stateful training, and no-std support
//! - **SafeTensors Format**: Industry-standard format for secure and efficient tensor serialization
//! - **PyTorch Compatibility**: Load PyTorch models directly into Burn with automatic weight transformation
//! - **Zero-Copy Loading**: Memory-mapped files and lazy tensor materialization for optimal performance
//! - **Flexible Filtering**: Load/save specific model subsets using regex, exact paths, or custom predicates
//! - **Tensor Remapping**: Rename tensors during load/save operations for framework compatibility
//! - **No-std Support**: Core functionality available in embedded and WASM environments
//!
//! ## Quick Start
//!
//! ### Basic Save and Load
//!
//! ```rust,ignore
//! use burn_store::{ModuleSnapshot, SafetensorsStore};
//!
//! // Save a model
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.save_into(&mut store)?;
//!
//! // Load a model
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.load_from(&mut store)?;
//! ```
//!
//! ### Loading PyTorch Models
//!
//! ```rust,ignore
//! use burn_store::PytorchStore;
//!
//! // Load PyTorch model (automatic weight transformation via PyTorchToBurnAdapter)
//! let mut store = PytorchStore::from_file("pytorch_model.pth")
//!     .with_top_level_key("state_dict")  // Access nested state dict if needed
//!     .allow_partial(true);               // Skip unknown tensors
//!
//! model.load_from(&mut store)?;
//! ```
//!
//! ### Filtering and Remapping
//!
//! ```rust,no_run
//! # use burn_store::SafetensorsStore;
//! // Save only specific layers with renaming
//! let mut store = SafetensorsStore::from_file("encoder.safetensors")
//!     .with_regex(r"^encoder\..*")                         // Filter: only encoder layers
//!     .with_key_remapping(r"^encoder\.", "transformer.")   // Rename: encoder.X -> transformer.X
//!     .metadata("subset", "encoder_only");
//!
//! // Use store with model.save_into(&mut store)?;
//! ```
//!
//! ## Core Components
//!
//! - [`ModuleSnapshot`]: Extension trait for Burn modules providing `collect()` and `apply()` methods
//! - [`BurnpackStore`]: Native Burn format with ParamId persistence for stateful training workflows
//! - [`SafetensorsStore`]: Primary storage implementation supporting the SafeTensors format
//! - [`PytorchStore`]: PyTorch model loader supporting .pth and .pt files
//! - [`PathFilter`]: Flexible filtering system for selective tensor loading/saving
//! - [`KeyRemapper`]: Advanced tensor name remapping with regex patterns
//! - [`ModuleAdapter`]: Framework adapters for cross-framework compatibility
//! - [`bridge`]: Conversions between burn-core tensors and the [`burn_pack::Tensor`] entries the stores move around
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and other std-only features (default)
//! - `safetensors`: Enables SafeTensors format support (default)
//! - `pytorch`: Enables loading PyTorch `.pt`/`.pth` files (default)
//! - `memmap`: Memory-maps files on load rather than reading them (default)

extern crate alloc;

mod adapter;
mod applier;
mod apply_result;
mod collector;
mod filter;
mod traits;

pub mod bridge;

pub use adapter::{
    BurnToPyTorchAdapter, ChainAdapter, FloatCastAdapter, HalfPrecisionAdapter, IdentityAdapter,
    ModuleAdapter, ModuleContext, PyTorchToBurnAdapter,
};
pub use applier::Applier;
pub use apply_result::{ApplyError, ApplyResult};
pub use collector::Collector;
pub use filter::PathFilter;
pub use traits::{ModuleSnapshot, ModuleStore};

#[cfg(feature = "std")]
mod keyremapper;
#[cfg(feature = "std")]
pub use keyremapper::{KeyRemapper, map_indices_contiguous};

/// Serde-based deserialization of nested values, used for importing model weights from external
/// formats (e.g. PyTorch's pickle `.pt` files).
#[cfg(feature = "pytorch")]
pub mod nested;

#[cfg(feature = "pytorch")]
pub mod pytorch;
#[cfg(feature = "pytorch")]
pub use pytorch::{PytorchStore, PytorchStoreError};

#[cfg(feature = "safetensors")]
mod safetensors;
#[cfg(feature = "safetensors")]
pub use safetensors::{SafetensorsStore, SafetensorsStoreError};

mod burnpack;
pub use burnpack::BurnpackStore;

/// The burnpack format crate, re-exported.
///
/// [`burn_pack::Tensor`] is burn-store's tensor-transport type: what
/// [`ModuleSnapshot::collect`] returns and what [`ModuleSnapshot::apply`] takes. A crate
/// holding some can drive a [`burn_pack::Writer`] itself instead of going through
/// [`BurnpackStore`], which is what you want when the tensors did not come from a
/// [`Module`](burn_core::module::Module) (weights read out of an ONNX file during codegen,
/// say). Nothing is materialized by collecting; each tensor is read back only when the writer
/// reaches it:
///
/// ```
/// use burn_store::burn_pack::{Bytes, Error, Tensor, Writer};
///
/// fn pack(tensors: Vec<Tensor>) -> Result<Bytes, Error> {
///     Writer::new(tensors).into_bytes()
/// }
/// ```
///
/// With burn-pack's `std` feature, `Writer::write_to_file` is the better choice for a large
/// model: it streams to disk instead of building the container in memory. This example uses
/// [`into_bytes`](burn_pack::Writer::into_bytes) so it compiles in no-std builds too.
///
/// Reach for this rather than depending on `burn-pack` directly, so the version always matches
/// the one burn-store was built against.
pub use burn_pack;
