//! Module persistence and serialization infrastructure.
//!
//! This module provides comprehensive functionality for persisting and loading Burn modules
//! and their tensor data. It supports bidirectional data flow with flexible filtering,
//! remapping, and efficient memory management through lazy materialization.

mod appliers;
mod base;
mod collectors;
mod filter;
mod keyremapper;
mod tensor_view;

#[cfg(test)]
mod test;

pub use appliers::{ApplyError, ApplyResult, TensorApplier};
pub use base::ModulePersist;
pub use base::ModulePersister;
pub use collectors::TensorViewCollector;
pub use filter::PathFilter;
pub use keyremapper::KeyRemapper;
pub use tensor_view::TensorView;
