//! PyTorch format support for burn-store.
//!
//! This module provides comprehensive support for loading PyTorch model files (.pth, .pt)
//! into Burn, with automatic weight transformation and flexible configuration options.
//!
//! ## Features
//!
//! - **Direct .pth/.pt file loading**: Load PyTorch checkpoint and state dict files
//! - **Automatic weight transformation**: `PyTorchToBurnAdapter` is applied by default:
//!   - Linear layer weights are automatically transposed
//!   - Normalization parameters are renamed (gamma → weight, beta → bias)
//!   - Conv2d weights maintain their format
//! - **Flexible filtering**: Load only specific layers or parameters
//! - **Key remapping**: Rename tensors during loading to match your model structure
//! - **Partial loading**: Continue even when some tensors are missing
//!
//! ## Example
//!
//! ```rust,ignore
//! use burn_store::PytorchStore;
//!
//! // Load a PyTorch model (PyTorchToBurnAdapter is applied automatically)
//! let mut store = PytorchStore::from_file("model.pth")
//!     .with_top_level_key("state_dict")              // Access nested state dict
//!     .with_regex(r"^encoder\..*")                   // Only load encoder layers
//!     .with_key_remapping(r"^fc\.", "linear.")       // Rename fc -> linear
//!     .allow_partial(true);                          // Skip missing tensors
//!
//! let mut model = MyModel::new(&device);
//! let result = model.apply_from(&mut store)?;
//!
//! println!("Loaded {} tensors", result.applied.len());
//! if !result.missing.is_empty() {
//!     println!("Missing tensors: {:?}", result.missing);
//! }
//! ```

pub mod lazy_data;
pub mod pickle_reader;
pub mod reader;
pub mod store;

#[cfg(test)]
pub mod tests;

// Main public interface
pub use reader::{PytorchError, PytorchReader};
pub use store::{PytorchStore, PytorchStoreError};
