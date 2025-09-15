#![cfg_attr(not(feature = "std"), no_std)]

//! Module storage and serialization infrastructure.
//!
//! This module provides comprehensive functionality for storing and loading Burn modules
//! and their tensor data. It supports bidirectional data flow with flexible filtering,
//! remapping, and efficient memory management through lazy materialization.

extern crate alloc;

mod adapter;
mod applier;
mod collector;
mod filter;
#[cfg(feature = "std")]
mod keyremapper;
pub mod safetensors;
mod tensor_snapshot;
mod traits;

pub use adapter::{BurnToPyTorchAdapter, IdentityAdapter, ModuleAdapter, PyTorchToBurnAdapter};
pub use applier::{Applier, ApplyError, ApplyResult};
pub use collector::Collector;
pub use filter::PathFilter;
#[cfg(feature = "std")]
pub use keyremapper::KeyRemapper;
pub use tensor_snapshot::TensorSnapshot;
pub use traits::ModuleSnapshot;
pub use traits::ModuleSnapshoter;

pub use safetensors::*;
