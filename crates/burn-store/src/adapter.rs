//! Module adapters for transforming tensors between different formats
//!
//! This module provides adapters that handle differences between PyTorch and Burn:
//! - Linear layer weight transposition
//! - Normalization parameter naming (weight/bias vs gamma/beta)

use crate::TensorSnapshot;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;

use burn_tensor::TensorData;

// Module type names as they appear in the container_type field
// These come from the Module derive macro which uses stringify! on the struct name
// Format: "Struct:TypeName" for user-defined structs
mod module_names {
    // Import the types to ensure they exist at compile time
    // If these types are renamed or moved, we'll get a compile error
    #[allow(unused_imports)]
    use burn_nn::{BatchNorm, GroupNorm, LayerNorm, Linear};

    // The actual string constants that match what the Module derive macro produces
    // The imports above ensure these types exist at compile-time
    pub const LINEAR: &str = "Struct:Linear";
    pub const BATCH_NORM: &str = "Struct:BatchNorm";
    pub const LAYER_NORM: &str = "Struct:LayerNorm";
    pub const GROUP_NORM: &str = "Struct:GroupNorm";
}

/// Trait for adapting tensor snapshots between different module formats
pub trait ModuleAdapter: Send + Sync {
    /// Adapt a tensor snapshot based on its container type and parameter name
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot;

    /// Get alternative parameter name to try during matching
    ///
    /// When looking for a parameter in a module, this method provides an alternative
    /// name to try if the direct name doesn't match. This enables matching parameters
    /// with different naming conventions (e.g., PyTorch's "weight" vs Burn's "gamma").
    ///
    /// # Arguments
    /// * `param_name` - The parameter name we're looking for
    /// * `container_type` - The type of container module (e.g., "BatchNorm")
    ///
    /// # Returns
    /// Alternative parameter name to try, or None if no alternative exists
    fn get_alternative_param_name(
        &self,
        _param_name: &str,
        _container_type: &str,
    ) -> Option<String> {
        None
    }

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
        adapt_pytorch_tensor(snapshot, PyTorchConversionDirection::PyTorchToBurn)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // For PyTorch->Burn: When looking for Burn names (gamma/beta), try PyTorch names (weight/bias)
        if is_normalization_layer(container_type) {
            burn_norm_param_to_pytorch(param_name).map(|s| s.to_string())
        } else {
            None
        }
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
        adapt_pytorch_tensor(snapshot, PyTorchConversionDirection::BurnToPyTorch)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // For Burn->PyTorch: When looking for PyTorch names (weight/bias), try Burn names (gamma/beta)
        if is_normalization_layer(container_type) {
            pytorch_norm_param_to_burn(param_name).map(|s| s.to_string())
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Direction of PyTorch conversion for parameter naming
#[derive(Debug, Clone, Copy)]
enum PyTorchConversionDirection {
    PyTorchToBurn,
    BurnToPyTorch,
}

/// Check if container type is a normalization layer
fn is_normalization_layer(container_type: &str) -> bool {
    matches!(
        container_type,
        module_names::BATCH_NORM | module_names::LAYER_NORM | module_names::GROUP_NORM
    )
}

/// Map PyTorch normalization parameter name to Burn
fn pytorch_norm_param_to_burn(param_name: &str) -> Option<&'static str> {
    match param_name {
        "weight" => Some("gamma"),
        "bias" => Some("beta"),
        _ => None,
    }
}

/// Map Burn normalization parameter name to PyTorch
fn burn_norm_param_to_pytorch(param_name: &str) -> Option<&'static str> {
    match param_name {
        "gamma" => Some("weight"),
        "beta" => Some("bias"),
        _ => None,
    }
}

/// Core tensor adaptation logic for PyTorch format conversions
fn adapt_pytorch_tensor(
    snapshot: &TensorSnapshot,
    direction: PyTorchConversionDirection,
) -> TensorSnapshot {
    // Extract path and parameter name
    let (path_stack, param_name) = match get_path_and_param(snapshot) {
        Some(result) => result,
        None => return snapshot.clone(),
    };

    // Get module type for matching (ignores Vec/Array wrappers)
    let module_type = match snapshot.module_type() {
        Some(mt) => mt,
        None => return snapshot.clone(), // No user-defined module found
    };

    // Linear: transpose weight (bidirectional - same operation both ways)
    if module_type == module_names::LINEAR && param_name == "weight" && snapshot.shape.len() == 2 {
        return transpose_2d_tensor(snapshot);
    }

    // Normalization layers: rename parameters based on direction
    if is_normalization_layer(&module_type) {
        let new_name = match direction {
            PyTorchConversionDirection::PyTorchToBurn => pytorch_norm_param_to_burn(param_name),
            PyTorchConversionDirection::BurnToPyTorch => burn_norm_param_to_pytorch(param_name),
        };

        if let Some(new_name) = new_name {
            return rename_parameter(snapshot, path_stack, new_name);
        }
    }

    snapshot.clone()
}

/// Extract path stack and parameter name from snapshot
fn get_path_and_param(snapshot: &TensorSnapshot) -> Option<(&[String], &str)> {
    let path_stack = snapshot.path_stack.as_ref()?;
    let param_name = path_stack.last()?.as_str();
    Some((path_stack.as_slice(), param_name))
}

/// Rename a parameter in the snapshot
fn rename_parameter(
    snapshot: &TensorSnapshot,
    path_stack: &[String],
    new_name: &str,
) -> TensorSnapshot {
    let mut new_path = path_stack.to_vec();
    *new_path.last_mut().unwrap() = new_name.to_string();

    TensorSnapshot::from_closure(
        snapshot.clone_data_fn(),
        snapshot.dtype,
        snapshot.shape.clone(),
        new_path,
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

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
        let data = original_data_fn()?;
        Ok(transpose_tensor_data(data))
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

/// Transpose tensor data (assumes 2D shape is already validated)
fn transpose_tensor_data(data: TensorData) -> TensorData {
    let shape = &data.shape;
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
            Rc::new(move || Ok(data.clone())),
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

        // Without container info, adapter returns unchanged for non-norm parameters
        let mut snapshot = create_test_snapshot("fc.weight", vec![10, 5], module_names::LINEAR);
        snapshot.container_stack = None;

        // Without container info, no transformation occurs for linear layers
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, vec![10, 5]); // No transposition without container info

        // Test a non-linear, non-norm parameter - should pass through unchanged
        let mut snapshot2 = create_test_snapshot("other.weight", vec![10, 5], "Struct:Other");
        snapshot2.container_stack = None;
        let adapted2 = adapter.adapt(&snapshot2);
        assert_eq!(adapted2.shape, vec![10, 5]); // No transposition
    }
}
