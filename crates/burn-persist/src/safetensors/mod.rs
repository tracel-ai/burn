//! SafeTensors format support for Burn deep learning framework.
//!
//! [SafeTensors](https://github.com/huggingface/safetensors) is a simple, safe, and efficient format
//! for storing and loading tensors. It provides fast zero-copy deserialization and strong safety
//! guarantees, making it ideal for production environments.
//!
//! # Features
//!
//! - **Fast Loading**: Zero-copy memory-mapped file support for instant tensor access
//! - **Safety**: Prevents arbitrary code execution during model loading
//! - **Efficiency**: Lazy deserialization only loads tensors when accessed
//! - **Filtering**: Load only specific tensors using path filters
//! - **Remapping**: Transform tensor names during load/save operations
//! - **Metadata**: Store and retrieve custom metadata alongside tensors
//! - **Cross-Platform**: Works on all platforms including no-std environments
//!
//! # Usage Examples
//!
//! ## Basic Save and Load
//!
//! ```rust,ignore
//! use burn::persist::safetensors::SafetensorsPersister;
//! use burn::persist::ModulePersist;
//!
//! // Save a model to a file
//! let mut persister = SafetensorsPersister::from_file("model.safetensors");
//! model.collect_to(&mut persister)?;
//!
//! // Load a model from a file  
//! let mut persister = SafetensorsPersister::from_file("model.safetensors");
//! let mut model = Model::new(&device);
//! model.apply_from(&mut persister)?;
//! ```
//!
//! ## Memory-Based Operations
//!
//! ```rust,ignore
//! use burn::persist::safetensors::SafetensorsPersister;
//! use burn::persist::ModulePersist;
//!
//! // Save to memory buffer
//! let mut persister = SafetensorsPersister::from_bytes(None);
//! model.collect_to(&mut persister)?;
//! let bytes = persister.get_bytes()?;
//!
//! // Load from memory buffer
//! let mut persister = SafetensorsPersister::from_bytes(Some(bytes));
//! let mut model = Model::new(&device);
//! model.apply_from(&mut persister)?;
//! ```
//!
//! ## Advanced Features
//!
//! ```rust,ignore
//! use burn::persist::{PathFilter, KeyRemapper, ModulePersist};
//! use burn::persist::safetensors::SafetensorsPersister;
//!
//! // Configure advanced options with method chaining
//! let mut persister = SafetensorsPersister::from_file("model.safetensors")
//!     // Only load encoder layers
//!     .filter(PathFilter::exact_paths(vec!["encoder.*"]))
//!     // Rename layers during loading
//!     .remap(KeyRemapper::new()
//!         .add_pattern(r"^old_name\.", "new_name.")?)
//!     // Add custom metadata
//!     .metadata("version", "1.0.0")
//!     .metadata("framework", "burn")
//!     // Allow partial loading (continue even if some tensors are missing)
//!     .allow_partial(true)
//!     // Disable validation for faster loading
//!     .validate(false);
//!
//! // Use the configured persister
//! model.collect_to(&mut persister)?;  // For saving
//! // or
//! model.apply_from(&mut persister)?;   // For loading
//! ```
//!
//! # Memory Mapping
//!
//! When the `memory-mapped` feature is enabled, SafeTensors automatically uses memory-mapped
//! files for extremely fast loading of large models:
//!
//! ```rust,ignore
//! // With the memory-mapped feature enabled, this automatically uses mmap
//! let mut persister = SafetensorsPersister::from_file("large_model.safetensors");
//! // Tensors are loaded lazily from the memory-mapped file
//! let mut model = Model::new(&device);
//! model.apply_from(&mut persister)?;
//! ```
//!
//! The memory-mapped feature provides:
//! - Zero-copy loading - tensors are read directly from disk
//! - Lazy loading - only accessed tensors are loaded into memory
//! - Shared memory - multiple processes can share the same mapped file
//!
//! # Working with Bytes
//!
//! For direct byte operations without files:
//!
//! ```rust,ignore
//! use burn::persist::ModulePersist;
//!
//! // Save to bytes
//! let mut persister = SafetensorsPersister::from_bytes(None);
//! model.collect_to(&mut persister)?;
//! let bytes = persister.get_bytes()?;
//!
//! // Load from bytes
//! let mut persister = SafetensorsPersister::from_bytes(Some(bytes));
//! let mut model = Model::new(&device);
//! model.apply_from(&mut persister)?;
//! ```
//!
//! # Format Details
//!
//! SafeTensors stores tensors in a simple binary format with:
//! - A JSON header containing tensor metadata and offsets
//! - Raw tensor data in little-endian byte order
//! - Optional user-defined metadata
//!
//! The format is designed to be:
//! - **Secure**: No pickle/unpickle, preventing code injection
//! - **Fast**: Direct memory mapping and zero-copy deserialization
//! - **Simple**: Minimal dependencies and straightforward implementation

mod persister;

pub use persister::{SafetensorsError, SafetensorsPersister};

#[cfg(test)]
mod tests;
