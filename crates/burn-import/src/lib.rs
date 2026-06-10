#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Import
//!
//! Model import and cross-framework interoperability for the Burn deep learning framework.
//!
//! This crate provides format loaders and stores that read external model formats into Burn
//! modules (and write some of them back out), built on top of the snapshot tooling in
//! [`burn_core::store`] and the burnpack format in [`burn_store`].
//!
//! ## Key Features
//!
//! - **SafeTensors Format**: Industry-standard format for secure and efficient tensor serialization
//! - **PyTorch Compatibility**: Load PyTorch `.pth`/`.pt` models directly into Burn
//! - **Burnpack Store**: A [`ModuleStore`](burn_core::store::ModuleStore) over the native burnpack format
//! - **Tensor Remapping**: Rename tensors during load/save operations for framework compatibility
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use burn_core::store::ModuleSnapshot;
//! use burn_import::SafetensorsStore;
//!
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.load_from(&mut store)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `std`: Enables file I/O and other std-only features (default)
//! - `safetensors`: Enables SafeTensors format support (default)
//! - `pytorch`: Enables PyTorch `.pth`/`.pt` loading (default)
//! - `burnpack`: Enables the native burnpack store (default)

extern crate alloc;

// Re-export the snapshot tooling whose canonical home is now [`burn_core::store`], so
// existing `crate::`-qualified paths within this crate (notably tests) keep resolving.
#[allow(unused_imports)]
pub(crate) use burn_core::store::{
    ApplyError, ApplyResult, Applier, BurnToPyTorchAdapter, ChainAdapter, Collector,
    HalfPrecisionAdapter, IdentityAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore, PathFilter,
    PyTorchToBurnAdapter, TensorSnapshot,
};

#[cfg(feature = "std")]
mod keyremapper;
#[cfg(feature = "std")]
pub use keyremapper::{KeyRemapper, map_indices_contiguous};

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
pub use burnpack::BurnpackStore;
