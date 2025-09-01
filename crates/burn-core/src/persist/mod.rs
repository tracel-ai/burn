//! Module persistence and serialization operations.
//!
//! This module provides functionality for exporting and persisting burn modules
//! and their associated tensors in various formats. It includes utilities for
//! creating lightweight tensor views that can be materialized on demand,
//! allowing for efficient memory usage during serialization operations.

mod base;
mod collectors;
/// Lightweight tensor view representations.
pub mod tensor_view;

pub use base::ModulePersist;
pub use collectors::TensorViewCollector;
pub use tensor_view::*;
