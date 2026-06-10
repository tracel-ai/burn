#![cfg_attr(not(feature = "std"), no_std)]

//! # Burn Store
//!
//! Model storage, import, and cross-framework interoperability for the Burn deep learning framework.
//!
//! This crate provides format loaders and stores that read external model formats into Burn
//! modules (and write some of them back out), built on top of the snapshot tooling in
//! [`burn_core::store`] and the burnpack format in [`burn_pack`].
//!
//! For convenience and easy migration, the snapshot tooling from [`burn_core::store`]
//! (e.g. [`ModuleSnapshot`], [`TensorSnapshot`], [`PathFilter`]) is re-exported here, so
//! `burn_store::ModuleSnapshot` and friends resolve directly.
//!
//! ## Key Features
//!
//! - **SafeTensors Format**: Industry-standard format for secure and efficient tensor serialization
//! - **PyTorch Compatibility**: Load PyTorch `.pth`/`.pt` models directly into Burn
//! - **Burnpack Store**: A [`ModuleStore`] over the native burnpack format
//! - **Tensor Remapping**: Rename tensors during load/save operations for framework compatibility
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use burn_store::{ModuleSnapshot, SafetensorsStore};
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

mod adapter;
mod applier;
mod apply_result;
mod collector;
mod filter;
mod tensor_snapshot;
mod traits;

pub use adapter::{
    BurnToPyTorchAdapter, ChainAdapter, HalfPrecisionAdapter, IdentityAdapter, ModuleAdapter,
    PyTorchToBurnAdapter,
};
pub use applier::Applier;
pub use apply_result::{ApplyError, ApplyResult};
pub use collector::Collector;
pub use filter::PathFilter;
pub use tensor_snapshot::{TensorSnapshot, TensorSnapshotError};
pub use traits::{ModuleSnapshot, ModuleStore};

// The non-generic record system lives in `burn-core`; re-export it so `burn_store::RecordNew`
// (and friends) resolve directly for an easy migration from the pre-split API.
pub use burn_core::store::{DTypePolicy, ModuleRecord, RecordError, RecordNew};

#[cfg(feature = "burnpack")]
mod bridge;

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
