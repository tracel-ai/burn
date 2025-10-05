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
//! - **Metadata**: Store and retrieve custom metadata alongside tensors (automatic `format`, `producer` and `version` metadata included)
//! - **Cross-Platform**: Works on all platforms including no-std environments
//!
//! # Usage Examples
//!
//! ## Basic Save and Load
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot};
//!
//! // Save a model to a file
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! model.collect_to(&mut store)?;
//!
//! // Load a model from a file
//! let mut store = SafetensorsStore::from_file("model.safetensors");
//! let mut model = Model::new(&device);
//! model.apply_from(&mut store)?;
//! ```
//!
//! ## Memory-Based Operations
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot};
//!
//! // Save to memory buffer
//! let mut store = SafetensorsStore::from_bytes(None);
//! model.collect_to(&mut store)?;
//! let bytes = store.get_bytes()?;
//!
//! // Load from memory buffer
//! let mut store = SafetensorsStore::from_bytes(Some(bytes));
//! let mut model = Model::new(&device);
//! model.apply_from(&mut store)?;
//! ```
//!
//! ## Advanced Features
//!
//! ### Filter Configuration with Builder Pattern
//!
//! ```rust,no_run
//! # use burn_store::SafetensorsStore;
//! // Filter with regex patterns (OR logic - matches any pattern)
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_regex(r"^encoder\..*")     // Match all encoder tensors
//!     .with_regex(r".*\.bias$");        // OR match any bias tensors
//!
//! // Filter with exact paths
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_full_path("encoder.weight")
//!     .with_full_path("encoder.bias")
//!     .with_full_paths(vec!["decoder.scale", "decoder.norm"]);
//!
//! // Custom filter logic with predicate
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_predicate(|path, _dtype| {
//!         // Only save layer weights (not biases)
//!         path.contains("layer") && path.ends_with("weight")
//!     });
//!
//! // Combine multiple filter methods
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_regex(r"^encoder\..*")           // All encoder tensors
//!     .with_full_path("decoder.scale")       // Plus specific decoder.scale
//!     .with_predicate(|path, _| {            // Plus any projection tensors
//!         path.contains("projection")
//!     });
//!
//! // Save or load all tensors (no filtering)
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .match_all();
//! ```
//!
//! ### Tensor Name Remapping
//!
//! Remap tensor names during load/save operations for compatibility between different frameworks:
//!
//! ```rust,no_run
//! # use burn_store::{SafetensorsStore, KeyRemapper};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Using builder pattern for common remapping patterns
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_key_remapping(r"^encoder\.", "transformer.encoder.")  // encoder.X -> transformer.encoder.X
//!     .with_key_remapping(r"\.gamma$", ".weight")                // X.gamma -> X.weight
//!     .with_key_remapping(r"\.beta$", ".bias");                  // X.beta -> X.bias
//!
//! // Or using a pre-configured KeyRemapper for complex transformations
//! let remapper = KeyRemapper::new()
//!     .add_pattern(r"^pytorch\.(.*)", "burn.$1")?           // pytorch.layer -> burn.layer
//!     .add_pattern(r"^(.*)\.running_mean$", "$1.mean")?     // layer.running_mean -> layer.mean
//!     .add_pattern(r"^(.*)\.running_var$", "$1.variance")?; // layer.running_var -> layer.variance
//!
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .remap(remapper);
//! # Ok(())
//! # }
//! ```
//!
//! ### Framework Adapters
//!
//! Use adapters for automatic framework-specific transformations:
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot, PyTorchToBurnAdapter, BurnToPyTorchAdapter};
//!
//! // Loading PyTorch model into Burn
//! let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
//!     .with_from_adapter(PyTorchToBurnAdapter)  // Transposes linear weights, renames norm params
//!     .allow_partial(true);                     // PyTorch models may have extra tensors
//!
//! let mut burn_model = Model::new(&device);
//! burn_model.apply_from(&mut store)?;
//!
//! // Saving Burn model for PyTorch
//! let mut store = SafetensorsStore::from_file("for_pytorch.safetensors")
//!     .with_to_adapter(BurnToPyTorchAdapter);   // Transposes weights back, renames for PyTorch
//!
//! burn_model.collect_to(&mut store)?;
//! ```
//!
//! ### Additional Configuration Options
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot};
//!
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     // Add custom metadata
//!     .metadata("version", "1.0.0")
//!     .metadata("producer", "burn")
//!     // Allow partial loading (continue even if some tensors are missing)
//!     .allow_partial(true)
//!     // Disable validation for faster loading
//!     .validate(false);
//!
//! // Use the configured store
//! model.collect_to(&mut store)?;  // For saving
//! // or
//! model.apply_from(&mut store)?;   // For loading
//! ```
//!
//! # Efficient Loading with SafeTensors
//!
//! SafeTensors provides efficient tensor loading through its zero-copy design:
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot};
//!
//! let mut store = SafetensorsStore::from_file("large_model.safetensors");
//! // Uses memory mapping (when available) for zero-copy access
//! // Falls back to buffered reading when mmap is not available
//! let mut model = Model::new(&device);
//! model.apply_from(&mut store)?;
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
//! use burn_store::SafetensorsStore;
//!
//! // Open a file - uses safetensors' efficient header reading
//! let store = SafetensorsStore::from_file("large_model.safetensors");
//!
//! // List all tensor names from the metadata
//! let tensor_names = store.list_tensors()?;
//! println!("Model contains {} tensors", tensor_names.len());
//!
//! // Get tensor metadata without loading tensor data
//! if let Some((shape, dtype)) = store.tensor_info("encoder.weight")? {
//!     println!("Encoder weight shape: {:?}, dtype: {:?}", shape, dtype);
//! }
//!
//! // Selectively load tensors - safetensors handles efficient access
//! let encoder_tensors = store.load_tensors(&[
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
//! let worker_tensors = store.load_tensors(&worker_layers)?;
//! ```
//!
//! # Builder Pattern API Reference
//!
//! The SafetensorsStore provides a fluent builder API for configuration:
//!
//! ## Filtering Methods
//!
//! - **`with_regex(pattern)`** - Add regex pattern to match tensor names (OR logic with multiple patterns)
//! - **`with_full_path(path)`** - Add exact tensor path to include
//! - **`with_full_paths(paths)`** - Add multiple exact tensor paths to include
//! - **`with_predicate(fn)`** - Add custom filter function `fn(&str, &str) -> bool`
//! - **`match_all()`** - Disable filtering, include all tensors
//!
//! ## Remapping Methods
//!
//! - **`with_key_remapping(from, to)`** - Add regex pattern to rename tensors
//! - **`remap(KeyRemapper)`** - Use a pre-configured KeyRemapper for complex transformations
//!
//! ## Adapter Methods
//!
//! - **`with_from_adapter(adapter)`** - Set adapter for loading (e.g., PyTorchToBurnAdapter)
//! - **`with_to_adapter(adapter)`** - Set adapter for saving (e.g., BurnToPyTorchAdapter)
//!
//! ## Configuration Methods
//!
//! - **`metadata(key, value)`** - Add custom metadata to saved files (in addition to automatic `format`, `producer` and `version`)
//! - **`allow_partial(bool)`** - Allow loading even if some tensors are missing
//! - **`validate(bool)`** - Enable/disable tensor validation during loading
//!
//! All methods return `Self` for chaining:
//!
//! ```rust,no_run
//! use burn_store::{SafetensorsStore, PyTorchToBurnAdapter};
//!
//! let store = SafetensorsStore::from_file("model.safetensors")
//!     .with_regex(r"^encoder\..*")
//!     .with_key_remapping(r"\.gamma$", ".weight")
//!     .with_from_adapter(PyTorchToBurnAdapter)
//!     .allow_partial(true)
//!     .metadata("version", "2.0");
//! ```
//!
//! # Working with Bytes
//!
//! For direct byte operations without files:
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot};
//!
//! // Save to bytes with filtering and remapping
//! let mut store = SafetensorsStore::from_bytes(None)
//!     .with_regex(r"^encoder\..*")                       // Only save encoder tensors
//!     .with_key_remapping(r"^encoder\.", "transformer.")  // Rename encoder.X -> transformer.X
//!     .metadata("subset", "encoder_only");
//! model.collect_to(&mut store)?;
//! let bytes = store.get_bytes()?;
//!
//! // Load from bytes (allow partial since we only saved encoder)
//! let mut store = SafetensorsStore::from_bytes(Some(bytes))
//!     .with_key_remapping(r"^transformer\.", "encoder.")  // Rename back: transformer.X -> encoder.X
//!     .allow_partial(true);
//! let mut model = Model::new(&device);
//! let result = model.apply_from(&mut store)?;
//! println!("Applied {} tensors", result.applied.len());
//! ```
//!
//! # Complete Example: PyTorch Model Migration
//!
//! Migrating a PyTorch model to Burn with filtering, remapping, and adapters:
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, ModuleSnapshot, PyTorchToBurnAdapter};
//!
//! // Load PyTorch model with all transformations
//! let mut store = SafetensorsStore::from_file("pytorch_model.safetensors")
//!     // Use PyTorch adapter for automatic transformations
//!     .with_from_adapter(PyTorchToBurnAdapter)
//!     // Only load transformer layers
//!     .with_regex(r"^transformer\..*")
//!     // Rename old layer names to new structure
//!     .with_key_remapping(r"^transformer\.h\.(\d+)\.", "transformer.layer$1.")
//!     // Skip unexpected tensors from PyTorch
//!     .allow_partial(true)
//!     // Add metadata about the conversion
//!     .metadata("source", "pytorch")
//!     .metadata("converted_by", "burn-store");
//!
//! let mut model = TransformerModel::new(&device);
//! let result = model.apply_from(&mut store)?;
//!
//! println!("Successfully loaded {} tensors", result.applied.len());
//! if !result.missing.is_empty() {
//!     println!("Missing tensors: {:?}", result.missing);
//! }
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

mod store;

pub use store::{SafetensorsStore, SafetensorsStoreError};

#[cfg(test)]
mod tests;
