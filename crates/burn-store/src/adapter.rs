//! Adapters for converting between different deep learning framework conventions.
//!
//! This module provides a flexible adapter system for handling differences between
//! deep learning frameworks when loading and saving models. Adapters enable seamless
//! conversion between frameworks like PyTorch and Burn by handling:
//!
//! - **Tensor transformations**: e.g., transposing linear layer weights
//! - **Parameter renaming**: e.g., converting between `weight`/`bias` and `gamma`/`beta`
//! - **Lazy evaluation**: All transformations are composed lazily for memory efficiency
//!
//! # Architecture
//!
//! The adapter system uses a single [`ModuleAdapter`] trait that can handle both:
//! - Loading: Converting from external formats to Burn format
//! - Saving: Converting from Burn format to external formats
//!
//! # Lazy Evaluation
//!
//! A key feature of this adapter system is **lazy evaluation**. When adapters transform
//! tensors, they don't immediately materialize the data. Instead, they compose transformations
//! as closures that are only executed when the tensor data is actually needed.
//!
//! ## Benefits of Lazy Evaluation:
//!
//! - **Memory Efficiency**: Large models aren't fully loaded into memory at once
//! - **Composability**: Multiple transformations can be chained without intermediate copies
//! - **Selective Loading**: Only tensors that pass filters and are needed are materialized
//! - **Fast Metadata Operations**: Renaming parameters doesn't touch actual tensor data
//!
//! ## How It Works:
//!
//! ```text
//! 1. SafeTensors file → TensorSnapshot with lazy closure
//! 2. Adapter wraps closure with transformation
//! 3. Data materialized only when module.apply() needs it
//! ```
//!
//! # Built-in Adapters
//!
//! - [`IdentityAdapter`]: No-op adapter that passes tensors through unchanged
//! - [`PyTorchToBurnAdapter`]: Converts PyTorch models to Burn format
//! - [`BurnToPyTorchAdapter`]: Converts Burn models to PyTorch format
//!
//! # Example
//!
//! ```rust,ignore
//! use burn_store::{SafetensorsStore, PyTorchToBurnAdapter};
//!
//! // Load a PyTorch model with automatic conversion
//! let mut store = SafetensorsStore::from_file("model.safetensors")
//!     .with_from_adapter(PyTorchToBurnAdapter);
//!
//! // The adapter will:
//! // 1. Transpose linear weights lazily (PyTorch: [out, in] → Burn: [in, out])
//! // 2. Rename normalization parameters (weight → gamma, bias → beta)
//! // 3. Only materialize tensors when actually needed
//! ```

use alloc::string::ToString;
use alloc::vec;
use burn_tensor::TensorData;

#[cfg(feature = "std")]
use crate::KeyRemapper;
use crate::TensorSnapshot;

/// Trait for adapting tensors between different framework conventions.
///
/// This trait allows conversion between different deep learning framework formats,
/// handling both loading and saving operations. Implementations can transform
/// tensor data, rename parameters, and modify shapes as needed.
pub trait ModuleAdapter {
    /// Adapt a tensor snapshot before saving it.
    ///
    /// This method can:
    /// - Transform tensor data (e.g., transpose weights)
    /// - Rename tensor keys (e.g., "gamma" -> "weight" for normalization layers)
    /// - Modify tensor shapes
    ///
    /// Returns `None` if the tensor should be skipped.
    fn adapt_tensor(&self, snapshot: &TensorSnapshot) -> Option<TensorSnapshot>;

    /// Get a key remapper for this adapter.
    ///
    /// This provides a convenient way to rename multiple keys systematically.
    #[cfg(feature = "std")]
    fn key_remapper(&self) -> Option<KeyRemapper> {
        None
    }
}

/// No-op adapter that passes tensors through unchanged.
#[derive(Debug, Clone, Default)]
pub struct IdentityAdapter;

impl ModuleAdapter for IdentityAdapter {
    fn adapt_tensor(&self, snapshot: &TensorSnapshot) -> Option<TensorSnapshot> {
        Some(snapshot.clone())
    }
}

/// Adapter for loading PyTorch models into Burn.
///
/// Handles differences between PyTorch and Burn conventions:
/// - Transposes linear layer weights (PyTorch: [out, in], Burn: [in, out])
/// - Renames normalization parameters (PyTorch: weight/bias, Burn: gamma/beta)
#[derive(Debug, Clone, Default)]
pub struct PyTorchToBurnAdapter;

impl ModuleAdapter for PyTorchToBurnAdapter {
    fn adapt_tensor(&self, snapshot: &TensorSnapshot) -> Option<TensorSnapshot> {
        let path = snapshot.full_path();

        // Check if this is a linear layer weight that needs transposing
        if is_linear_weight(&path) {
            return Some(transpose_2d_tensor(snapshot));
        }

        // Check if this needs parameter renaming (normalization layers)
        if let Some(renamed_snapshot) = rename_normalization_params_to_burn(snapshot) {
            return Some(renamed_snapshot);
        }

        // Pass through unchanged
        Some(snapshot.clone())
    }

    #[cfg(feature = "std")]
    fn key_remapper(&self) -> Option<KeyRemapper> {
        // For now, we handle renaming in adapt_tensor directly
        // This could be extended to use KeyRemapper if needed
        None
    }
}

/// Adapter for saving Burn models to PyTorch format.
///
/// Handles differences between Burn and PyTorch conventions:
/// - Transposes linear layer weights (Burn: [in, out], PyTorch: [out, in])
/// - Renames normalization parameters (Burn: gamma/beta, PyTorch: weight/bias)
#[derive(Debug, Clone, Default)]
pub struct BurnToPyTorchAdapter;

impl ModuleAdapter for BurnToPyTorchAdapter {
    fn adapt_tensor(&self, snapshot: &TensorSnapshot) -> Option<TensorSnapshot> {
        let path = snapshot.full_path();

        // Check if this is a linear layer weight that needs transposing
        if is_linear_weight(&path) {
            return Some(transpose_2d_tensor(snapshot));
        }

        // Check if this needs parameter renaming (normalization layers)
        if let Some(renamed_snapshot) = rename_normalization_params_to_pytorch(snapshot) {
            return Some(renamed_snapshot);
        }

        // Pass through unchanged
        Some(snapshot.clone())
    }

    #[cfg(feature = "std")]
    fn key_remapper(&self) -> Option<KeyRemapper> {
        // For now, we handle renaming in adapt_tensor directly
        // This could be extended to use KeyRemapper if needed
        None
    }
}

// Helper functions

fn is_linear_weight(path: &str) -> bool {
    // Check if this is a weight tensor from a linear/dense layer
    // This is a simplified check - you might need to make it more sophisticated
    (path.contains("linear") || path.contains("fc") || path.contains("dense"))
        && path.ends_with(".weight")
}

fn transpose_2d_tensor(snapshot: &TensorSnapshot) -> TensorSnapshot {
    // Only transpose 2D tensors
    if snapshot.shape.len() != 2 {
        return snapshot.clone();
    }

    // Clone the original data function for lazy composition
    let original_data_fn = snapshot.clone_data_fn();
    let dtype = snapshot.dtype;

    // Create transposed shape
    let transposed_shape = vec![snapshot.shape[1], snapshot.shape[0]];

    // Create a new lazy closure that transposes when called
    let transposed_data_fn = alloc::rc::Rc::new(move || {
        let data = original_data_fn();
        transpose_tensor_data(data)
    });

    // Create a new snapshot with lazy transposition
    TensorSnapshot::from_closure(
        transposed_data_fn,
        dtype,
        transposed_shape,
        snapshot.path_stack.clone().unwrap_or_default(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

fn transpose_tensor_data(data: TensorData) -> TensorData {
    // Get the shape
    let shape = data.shape.clone();
    if shape.len() != 2 {
        return data; // Only transpose 2D tensors
    }

    let rows = shape[0];
    let cols = shape[1];

    // Create transposed shape
    let transposed_shape = vec![cols, rows];

    // Transpose the data based on dtype
    match data.dtype {
        burn_tensor::DType::F32 => {
            let values = data.to_vec::<f32>().unwrap();
            let mut transposed = vec![0.0f32; values.len()];

            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = values[i * cols + j];
                }
            }

            TensorData::new(transposed, transposed_shape)
        }
        burn_tensor::DType::F64 => {
            let values = data.to_vec::<f64>().unwrap();
            let mut transposed = vec![0.0f64; values.len()];

            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = values[i * cols + j];
                }
            }

            TensorData::new(transposed, transposed_shape)
        }
        _ => data, // For other dtypes, return unchanged
    }
}

fn rename_normalization_params_to_burn(snapshot: &TensorSnapshot) -> Option<TensorSnapshot> {
    let path = snapshot.full_path();

    // Check if this is a normalization layer parameter
    if !path.contains("norm") && !path.contains("bn") {
        return None;
    }

    // Rename weight -> gamma, bias -> beta
    let mut path_stack = snapshot.path_stack.clone().unwrap_or_default();
    if let Some(last) = path_stack.last_mut() {
        if last == "weight" {
            *last = "gamma".to_string();
        } else if last == "bias" {
            *last = "beta".to_string();
        } else {
            return None; // No renaming needed
        }
    } else {
        return None;
    }

    // Create a new snapshot with renamed path - keep the same lazy data function
    Some(TensorSnapshot::from_closure(
        snapshot.clone_data_fn(), // Reuse the same lazy closure
        snapshot.dtype,
        snapshot.shape.clone(),
        path_stack,
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    ))
}

fn rename_normalization_params_to_pytorch(snapshot: &TensorSnapshot) -> Option<TensorSnapshot> {
    let path = snapshot.full_path();

    // Check if this is a normalization layer parameter
    if !path.contains("norm") && !path.contains("bn") {
        return None;
    }

    // Rename gamma -> weight, beta -> bias
    let mut path_stack = snapshot.path_stack.clone().unwrap_or_default();
    if let Some(last) = path_stack.last_mut() {
        if last == "gamma" {
            *last = "weight".to_string();
        } else if last == "beta" {
            *last = "bias".to_string();
        } else {
            return None; // No renaming needed
        }
    } else {
        return None;
    }

    // Create a new snapshot with renamed path - keep the same lazy data function
    Some(TensorSnapshot::from_closure(
        snapshot.clone_data_fn(), // Reuse the same lazy closure
        snapshot.dtype,
        snapshot.shape.clone(),
        path_stack,
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::module::ParamId;

    #[test]
    fn test_identity_adapter() {
        let adapter = IdentityAdapter;
        let snapshot = create_test_snapshot("model.weight", vec![2, 3]);

        let adapted = ModuleAdapter::adapt_tensor(&adapter, &snapshot);
        assert!(adapted.is_some());

        let adapted = adapted.unwrap();
        assert_eq!(adapted.full_path(), "model.weight");
        assert_eq!(adapted.shape, vec![2, 3]);
    }

    #[test]
    fn test_pytorch_to_burn_linear_weight() {
        let adapter = PyTorchToBurnAdapter;
        let snapshot = create_test_snapshot("model.linear.weight", vec![3, 2]);

        let adapted = adapter.adapt_tensor(&snapshot);
        assert!(adapted.is_some());

        let adapted = adapted.unwrap();
        assert_eq!(adapted.full_path(), "model.linear.weight");
        assert_eq!(adapted.shape, vec![2, 3]); // Transposed
    }

    #[test]
    fn test_pytorch_to_burn_norm_params() {
        let adapter = PyTorchToBurnAdapter;

        // Test weight -> gamma
        let snapshot = create_test_snapshot("model.norm.weight", vec![10]);
        let adapted = adapter.adapt_tensor(&snapshot).unwrap();
        assert_eq!(adapted.full_path(), "model.norm.gamma");

        // Test bias -> beta
        let snapshot = create_test_snapshot("model.norm.bias", vec![10]);
        let adapted = adapter.adapt_tensor(&snapshot).unwrap();
        assert_eq!(adapted.full_path(), "model.norm.beta");
    }

    #[test]
    fn test_burn_to_pytorch_linear_weight() {
        let adapter = BurnToPyTorchAdapter;
        let snapshot = create_test_snapshot("model.linear.weight", vec![2, 3]);

        let adapted = adapter.adapt_tensor(&snapshot);
        assert!(adapted.is_some());

        let adapted = adapted.unwrap();
        assert_eq!(adapted.full_path(), "model.linear.weight");
        assert_eq!(adapted.shape, vec![3, 2]); // Transposed
    }

    #[test]
    fn test_burn_to_pytorch_norm_params() {
        let adapter = BurnToPyTorchAdapter;

        // Test gamma -> weight
        let snapshot = create_test_snapshot("model.norm.gamma", vec![10]);
        let adapted = adapter.adapt_tensor(&snapshot).unwrap();
        assert_eq!(adapted.full_path(), "model.norm.weight");

        // Test beta -> bias
        let snapshot = create_test_snapshot("model.norm.beta", vec![10]);
        let adapted = adapter.adapt_tensor(&snapshot).unwrap();
        assert_eq!(adapted.full_path(), "model.norm.bias");
    }

    fn create_test_snapshot(path: &str, shape: Vec<usize>) -> TensorSnapshot {
        let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
        let data = match shape.len() {
            1 => TensorData::new(vec![1.0f32; shape[0]], shape.clone()),
            2 => TensorData::new(vec![1.0f32; shape[0] * shape[1]], shape.clone()),
            _ => TensorData::new(vec![1.0f32], vec![1]),
        };

        TensorSnapshot::from_data(data, path_parts, vec!["Module".to_string()], ParamId::new())
    }
}
