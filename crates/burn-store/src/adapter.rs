//! Module adapters for transforming tensors between different formats
//!
//! This module provides adapters that handle differences between PyTorch and Burn:
//! - Linear layer weight transposition
//! - Normalization parameter naming (weight/bias vs gamma/beta)

use crate::TensorSnapshot;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;

use burn_tensor::TensorData;

// Module type names as they appear in the container_type field
// These come from the Module derive macro which uses stringify! on the struct name
mod module_names {
    // Import the types to ensure they exist at compile time
    // If these types are renamed or moved, we'll get a compile error
    #[allow(unused_imports)]
    use burn_nn::{BatchNorm, GroupNorm, LayerNorm, Linear}; // RENAME CONSTANTS TOO

    // The actual string constants that match what the Module derive macro produces
    // The imports above ensure these types exist at compile-time
    pub const LINEAR: &str = "Linear";
    pub const BATCH_NORM: &str = "BatchNorm";
    pub const LAYER_NORM: &str = "LayerNorm";
    pub const GROUP_NORM: &str = "GroupNorm";
}

/// Trait for adapting tensor snapshots between different module formats
pub trait ModuleAdapter: Send + Sync {
    /// Adapt a tensor snapshot based on its container type and parameter name
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot;

    /// Clone the adapter into a boxed trait object
    fn clone_box(&self) -> Box<dyn ModuleAdapter>;
}

impl Clone for Box<dyn ModuleAdapter> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Identity adapter that passes tensors through unchanged
#[derive(Debug, Clone, Default)]
pub struct IdentityAdapter;

impl ModuleAdapter for IdentityAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter for converting from PyTorch format to Burn format
///
/// Handles:
/// - Linear layer weight transposition (PyTorch: [out, in] → Burn: [in, out])
/// - Normalization parameter renaming (weight → gamma, bias → beta)
#[derive(Debug, Clone, Default)]
pub struct PyTorchToBurnAdapter;

impl ModuleAdapter for PyTorchToBurnAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // Get parameter name (last path item)
        let path_stack = match snapshot.path_stack.as_ref() {
            Some(stack) => stack,
            None => return snapshot.clone(),
        };

        let param_name = match path_stack.last() {
            Some(name) => name,
            None => return snapshot.clone(),
        };

        // Check container type if available
        if let Some(container_stack) = snapshot.container_stack.as_ref()
            && let Some(container_type) = container_stack.last()
        {
            match container_type.as_str() {
                // Linear: transpose weight
                module_names::LINEAR if param_name == "weight" && snapshot.shape.len() == 2 => {
                    return transpose_2d_tensor(snapshot);
                }
                // Normalization layers: rename weight->gamma, bias->beta
                module_names::BATCH_NORM | module_names::LAYER_NORM | module_names::GROUP_NORM => {
                    let new_name = match param_name.as_str() {
                        "weight" => "gamma",
                        "bias" => "beta",
                        _ => return snapshot.clone(),
                    };

                    let mut new_path = path_stack.clone();
                    let last_idx = new_path.len() - 1;
                    new_path[last_idx] = new_name.to_string();
                    return TensorSnapshot::from_closure(
                        snapshot.clone_data_fn(),
                        snapshot.dtype,
                        snapshot.shape.clone(),
                        new_path,
                        container_stack.clone(),
                        snapshot.tensor_id.unwrap_or_default(),
                    );
                }
                _ => {}
            }
        }

        // No transformation needed
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter for converting from Burn format to PyTorch format
///
/// Handles:
/// - Linear layer weight transposition (Burn: [in, out] → PyTorch: [out, in])
/// - Normalization parameter renaming (gamma → weight, beta → bias)
#[derive(Debug, Clone, Default)]
pub struct BurnToPyTorchAdapter;

impl ModuleAdapter for BurnToPyTorchAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // Get parameter name (last path item)
        let path_stack = match snapshot.path_stack.as_ref() {
            Some(stack) => stack,
            None => return snapshot.clone(),
        };

        let param_name = match path_stack.last() {
            Some(name) => name,
            None => return snapshot.clone(),
        };

        // Check container type if available
        if let Some(container_stack) = snapshot.container_stack.as_ref()
            && let Some(container_type) = container_stack.last()
        {
            match container_type.as_str() {
                // Linear: transpose weight
                module_names::LINEAR if param_name == "weight" && snapshot.shape.len() == 2 => {
                    return transpose_2d_tensor(snapshot);
                }
                // Normalization layers: rename gamma->weight, beta->bias
                module_names::BATCH_NORM | module_names::LAYER_NORM | module_names::GROUP_NORM => {
                    match param_name.as_str() {
                        "gamma" | "beta" => {
                            let new_name = match param_name.as_str() {
                                "gamma" => "weight",
                                "beta" => "bias",
                                _ => return snapshot.clone(),
                            };

                            let mut new_path = path_stack.clone();
                            let last_idx = new_path.len() - 1;
                            new_path[last_idx] = new_name.to_string();
                            return TensorSnapshot::from_closure(
                                snapshot.clone_data_fn(),
                                snapshot.dtype,
                                snapshot.shape.clone(),
                                new_path,
                                container_stack.clone(),
                                snapshot.tensor_id.unwrap_or_default(),
                            );
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // No transformation needed
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

// Helper functions

/// Transpose a 2D tensor
fn transpose_2d_tensor(snapshot: &TensorSnapshot) -> TensorSnapshot {
    if snapshot.shape.len() != 2 {
        return snapshot.clone();
    }

    let original_data_fn = snapshot.clone_data_fn();
    let dtype = snapshot.dtype;
    let transposed_shape = vec![snapshot.shape[1], snapshot.shape[0]];

    // Create a lazy closure that transposes when called
    let transposed_data_fn = Rc::new(move || {
        let data = original_data_fn();
        transpose_tensor_data(data)
    });

    TensorSnapshot::from_closure(
        transposed_data_fn,
        dtype,
        transposed_shape,
        snapshot.path_stack.clone().unwrap_or_default(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

/// Transpose tensor data
fn transpose_tensor_data(data: TensorData) -> TensorData {
    let shape = data.shape.clone();
    if shape.len() != 2 {
        return data;
    }

    let rows = shape[0];
    let cols = shape[1];
    let transposed_shape = vec![cols, rows];

    // Get the raw bytes and element size
    let bytes = data.as_bytes();
    let element_size = data.dtype.size();

    // Create a new buffer for transposed data
    let mut transposed_bytes = vec![0u8; bytes.len()];

    // Transpose at the byte level - works for any data type
    for i in 0..rows {
        for j in 0..cols {
            let src_idx = (i * cols + j) * element_size;
            let dst_idx = (j * rows + i) * element_size;

            // Copy the bytes for this element
            transposed_bytes[dst_idx..dst_idx + element_size]
                .copy_from_slice(&bytes[src_idx..src_idx + element_size]);
        }
    }

    // Create new TensorData from transposed bytes
    TensorData::from_bytes_vec(transposed_bytes, transposed_shape, data.dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use burn_tensor::{DType, TensorData};

    fn create_test_snapshot(path: &str, shape: Vec<usize>, container_type: &str) -> TensorSnapshot {
        let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
        let values = vec![1.0f32; shape.iter().product()];
        let data = TensorData::new(values, shape.clone());

        TensorSnapshot::from_closure(
            Rc::new(move || data.clone()),
            DType::F32,
            shape,
            path_parts,
            vec![container_type.to_string()],
            burn_core::module::ParamId::new(),
        )
    }

    #[test]
    fn test_pytorch_to_burn_linear_weight() {
        let adapter = PyTorchToBurnAdapter;

        // Linear layer weight should be transposed
        let snapshot = create_test_snapshot("fc.weight", vec![10, 5], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, vec![5, 10]);

        // Linear layer bias should not be transposed
        let snapshot = create_test_snapshot("fc.bias", vec![10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, vec![10]);
    }

    #[test]
    fn test_pytorch_to_burn_norm_params() {
        let adapter = PyTorchToBurnAdapter;

        // BatchNorm weight -> gamma
        let snapshot = create_test_snapshot("norm.weight", vec![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.gamma");

        // BatchNorm bias -> beta
        let snapshot = create_test_snapshot("norm.bias", vec![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.beta");
    }

    #[test]
    fn test_burn_to_pytorch_linear_weight() {
        let adapter = BurnToPyTorchAdapter;

        // Linear layer weight should be transposed
        let snapshot = create_test_snapshot("fc.weight", vec![5, 10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, vec![10, 5]);
    }

    #[test]
    fn test_burn_to_pytorch_norm_params() {
        let adapter = BurnToPyTorchAdapter;

        // BatchNorm gamma -> weight
        let snapshot = create_test_snapshot("norm.gamma", vec![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.weight");

        // BatchNorm beta -> bias
        let snapshot = create_test_snapshot("norm.beta", vec![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.bias");
    }

    #[test]
    fn test_transpose_different_dtypes() {
        // Test that transpose works for different data types

        // Test with F32
        let f32_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = transpose_tensor_data(f32_data);
        assert_eq!(transposed.shape, vec![3, 2]);
        let values = transposed.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test with I32
        let i32_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], vec![2, 3]);
        let transposed = transpose_tensor_data(i32_data);
        assert_eq!(transposed.shape, vec![3, 2]);
        let values = transposed.to_vec::<i32>().unwrap();
        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

        // Test with F64
        let f64_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let transposed = transpose_tensor_data(f64_data);
        assert_eq!(transposed.shape, vec![2, 2]);
        let values = transposed.to_vec::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_no_container_info() {
        let adapter = PyTorchToBurnAdapter;

        // Without container info, adapter returns unchanged
        let mut snapshot = create_test_snapshot("fc.weight", vec![10, 5], module_names::LINEAR);
        snapshot.container_stack = None;

        // Without container info, no transformation occurs
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, vec![10, 5]); // No transposition without container info

        // Test a non-linear, non-norm parameter - should pass through unchanged
        let mut snapshot2 = create_test_snapshot("other.weight", vec![10, 5], "Other");
        snapshot2.container_stack = None;
        let adapted2 = adapter.adapt(&snapshot2);
        assert_eq!(adapted2.shape, vec![10, 5]); // No transposition
    }
}
