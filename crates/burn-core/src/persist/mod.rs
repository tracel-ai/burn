//! Module persistence and serialization infrastructure.
//!
//! This module provides comprehensive functionality for persisting and loading Burn modules
//! and their tensor data. It supports bidirectional data flow with flexible filtering,
//! remapping, and efficient memory management through lazy materialization.
//!
//! # Features
//!
//! - **Collection**: Extract tensor data from modules with regex-based filtering
//! - **Apply**: Apply tensor data to modules with validation and error handling
//! - **TensorView**: Lightweight tensor representations that materialize data on demand
//! - **Filtering**: Regex patterns and predicates for selective tensor operations
//! - **Remapping**: Transform tensor paths during apply operations for framework interoperability
//!
//! # Example Usage
//!
//! ```ignore
//! use burn_core::persist::ModulePersist;
//!
//! // Collect tensors from a model
//! let tensor_views = model.collect();
//!
//! // Apply to another model with filtering
//! let result = other_model.apply_filtered(
//!     tensor_views,
//!     &[r"^encoder\..*"]  // Only apply encoder tensors
//! )?;
//! ```

mod appliers;
mod base;
mod collectors;
mod filter;
mod tensor_view;

pub use appliers::{ImportError, ImportResult, TensorApplier};
pub use base::ModulePersist;
pub use collectors::TensorViewCollector;
pub use tensor_view::TensorView;
