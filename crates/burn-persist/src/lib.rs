#![cfg_attr(not(feature = "std"), no_std)]

//! Module persistence and serialization infrastructure.
//!
//! This module provides comprehensive functionality for persisting and loading Burn modules
//! and their tensor data. It supports bidirectional data flow with flexible filtering,
//! remapping, and efficient memory management through lazy materialization.

extern crate alloc;

mod adapter;
mod applier;
mod collector;
mod filter;
mod keyremapper;
pub mod safetensors;
mod tensor_snapshot;
mod traits;

pub use adapter::{Adapter, BurnToPyTorchAdapter, IdentityAdapter, PyTorchToBurnAdapter};
pub use applier::{ApplyError, ApplyResult, TensorApplier};
pub use collector::TensorSnapshotCollector;
pub use filter::PathFilter;
pub use keyremapper::KeyRemapper;
pub use tensor_snapshot::TensorSnapshot;
pub use traits::ModulePersist;
pub use traits::ModulePersister;

pub use safetensors::*;
