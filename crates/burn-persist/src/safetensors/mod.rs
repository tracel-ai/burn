//! SafeTensors format support for Burn deep learning framework.
//!
//! [SafeTensors](https://github.com/huggingface/safetensors) is a simple, safe, and efficient format
//! for storing and loading tensors. It provides fast zero-copy deserialization and strong safety
//! guarantees, making it ideal for production environments.
//!
//! # Features
//!
//! - **Fast Loading**: Zero-copy tensor access using safetensors' built-in mechanisms
//! - **Safety**: Prevents arbitrary code execution during model loading
//! - **Efficiency**: Memory-mapped files enable lazy loading without reading entire file
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
//! use burn_persist::safetensors::SafetensorsPersister;
//! use burn_persist::ModulePersist;
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
//! use burn_persist::safetensors::SafetensorsPersister;
//! use burn_persist::ModulePersist;
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
//! use burn_persist::{PathFilter, KeyRemapper, ModulePersist};
//! use burn_persist::safetensors::SafetensorsPersister;
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
//! # Efficient Loading with SafeTensors
//!
//! SafeTensors provides efficient tensor loading through its zero-copy design:
//!
//! ```rust,ignore
//! let mut persister = SafetensorsPersister::from_file("large_model.safetensors");
//! // Uses memory mapping (when available) for zero-copy access
//! // Falls back to buffered reading when mmap is not available
//! let mut model = Model::new(&device);
//! model.apply_from(&mut persister)?;
//! ```
//!
//! The safetensors approach provides:
//! - Zero-copy views - tensors are accessed directly from the mapped file
//! - Lazy loading - only accessed tensors are materialized
//! - Efficient memory usage - no unnecessary data duplication
//!
//! # Lazy Loading and Inspection
//!
//! SafeTensors provides efficient inspection and selective loading through its
//! zero-copy design and built-in metadata handling:
//!
//! ```rust,ignore
//! use burn_persist::safetensors::SafetensorsPersister;
//!
//! // Open a file - uses safetensors' efficient header reading
//! let persister = SafetensorsPersister::from_file("large_model.safetensors");
//!
//! // List all tensor names from the metadata
//! let tensor_names = persister.list_tensors()?;
//! println!("Model contains {} tensors", tensor_names.len());
//!
//! // Get tensor metadata without loading tensor data
//! if let Some((shape, dtype)) = persister.tensor_info("encoder.weight")? {
//!     println!("Encoder weight shape: {:?}, dtype: {:?}", shape, dtype);
//! }
//!
//! // Selectively load tensors - safetensors handles efficient access
//! let encoder_tensors = persister.load_tensors(&[
//!     "encoder.weight",
//!     "encoder.bias",
//!     "encoder.norm"
//! ])?;
//!
//! // Distributed loading: each worker loads only its assigned layers
//! // SafeTensors' zero-copy views ensure minimal memory usage
//! let worker_layers = match worker_id {
//!     0 => vec!["encoder.layer1", "encoder.layer2"],
//!     1 => vec!["encoder.layer3", "encoder.layer4"],
//!     2 => vec!["decoder.layer1", "decoder.layer2"],
//!     _ => vec!["head.weight", "head.bias"],
//! };
//! let worker_tensors = persister.load_tensors(&worker_layers)?;
//! ```
//!
//! # Working with Bytes
//!
//! For direct byte operations without files:
//!
//! ```rust,ignore
//! use burn_persist::ModulePersist;
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
//! SafeTensors uses a simple binary format:
//! - **8 bytes**: Header size (unsigned little-endian 64-bit integer)
//! - **N bytes**: JSON header with tensor metadata
//!   - Contains: `{"tensor_name": {"dtype": "F32", "shape": [1, 2, 3], "data_offsets": [start, end]}, ...}`
//!   - Special key `__metadata__` for user-defined string metadata
//! - **Rest**: Raw tensor data (referenced by offsets in header)
//!
//! The format enables:
//! - **Secure loading**: No code execution, just data
//! - **Efficient access**: Use offsets to read only needed tensors
//! - **Simple parsing**: Standard JSON header with fixed structure

mod persister;

pub use persister::{SafetensorsError, SafetensorsPersister};

#[cfg(test)]
mod tests;
