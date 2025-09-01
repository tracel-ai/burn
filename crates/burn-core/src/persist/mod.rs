//! Module persistence and serialization infrastructure.
//!
//! This module provides comprehensive functionality for persisting and loading Burn modules
//! and their tensor data. It supports bidirectional data flow with flexible filtering,
//! remapping, and efficient memory management through lazy materialization.
//!
//! # Features
//!
//! - **Export**: Extract tensor data from modules with regex-based filtering
//! - **Import**: Load tensor data into modules with validation and error handling
//! - **TensorView**: Lightweight tensor representations that materialize data on demand
//! - **Filtering**: Regex patterns and predicates for selective tensor operations
//! - **Remapping**: Transform tensor paths during import/export for framework interoperability
//!
//! # Example Usage
//!
//! ```ignore
//! use burn_core::persist::{ModuleExport, ModuleImport};
//!
//! // Export tensors from a model
//! let tensor_views = model.export_tensor_views();
//!
//! // Import into another model with filtering
//! let result = other_model.import_tensor_views_filtered(
//!     tensor_views,
//!     &[r"^encoder\..*"]  // Only import encoder tensors
//! )?;
//! ```

mod base;
mod collectors;
/// Lightweight tensor view representations.
pub mod tensor_view;

pub use base::ModulePersist;
pub use collectors::TensorViewCollector;
/// Module export functionality for extracting tensor data.
/// Module export functionality for extracting tensor data from modules.
///
/// Provides methods to export tensor views with optional filtering using regex patterns
/// or custom predicates. Exported views are lazy and only materialize data when needed.
pub mod export;

/// Module import functionality for loading tensor data into modules.
///
/// Provides methods to import tensor views with validation, filtering, and path remapping.
/// Supports partial imports and provides detailed error reporting for mismatches.
pub mod import;

/// Lightweight tensor view representations for efficient data handling.
///
/// TensorView provides a lazy representation of tensor data that can be materialized
/// on demand, reducing memory usage during persistence operations.
pub mod tensor_view;

pub use export::*;
pub use import::*;
pub use tensor_view::*;
