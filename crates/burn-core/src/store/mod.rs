//! Module snapshot tooling and the non-generic record system.
//!
//! This module provides the machinery to collect a module's parameters into lazy
//! [`TensorSnapshot`]s, apply snapshots back onto a module, and serialize them via
//! the burnpack format (see [`burn_store`]) through the non-generic [`RecordNew`].
//!
//! It is built *alongside* the legacy [`crate::record`] system.

mod adapter;
mod applier;
mod apply_result;
pub mod bridge;
mod collector;
mod filter;
mod record;
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
pub use record::{DTypePolicy, RecordError, RecordNew};
pub use tensor_snapshot::{TensorSnapshot, TensorSnapshotError};
pub use traits::{ModuleSnapshot, ModuleStore};
