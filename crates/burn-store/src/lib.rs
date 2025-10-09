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
//! - [`SafetensorsStore`]: Primary storage implementation supporting the SafeTensors format
//! - [`PytorchStore`]: PyTorch model loader supporting .pth and .pt files
//! - [`PathFilter`]: Flexible filtering system for selective tensor loading/saving
//! - [`KeyRemapper`]: Advanced tensor name remapping with regex patterns
//! - [`ModuleAdapter`]: Framework adapters for cross-framework compatibility
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and other std-only features (default)
//! - `safetensors`: Enables SafeTensors format support (default)

extern crate alloc;

mod adapter;
mod applier;
mod collector;
mod filter;
mod tensor_snapshot;
mod traits;

pub use adapter::{BurnToPyTorchAdapter, IdentityAdapter, ModuleAdapter, PyTorchToBurnAdapter};
pub use applier::{Applier, ApplyError, ApplyResult};
pub use collector::Collector;
pub use filter::PathFilter;
pub use tensor_snapshot::{TensorSnapshot, TensorSnapshotError};
pub use traits::{ModuleSnapshot, ModuleSnapshoter};

#[cfg(feature = "std")]
mod keyremapper;
#[cfg(feature = "std")]
pub use keyremapper::KeyRemapper;

#[cfg(feature = "pytorch")]
pub mod pytorch;
#[cfg(feature = "pytorch")]
pub use pytorch::{PytorchStore, PytorchStoreError};

#[cfg(feature = "safetensors")]
mod safetensors;
#[cfg(feature = "safetensors")]
pub use safetensors::{SafetensorsStore, SafetensorsStoreError};

#[cfg(feature = "burnpack")]
mod burnpack;
#[cfg(feature = "burnpack")]
pub use burnpack::store::BurnpackStore;
