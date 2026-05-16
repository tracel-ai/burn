//! Module adapters for transforming tensor snapshots during save/load
//!
//! This module provides adapters for:
//! - PyTorch/Burn format conversion (weight transposition, parameter renaming)
//! - Mixed-precision storage (F32/F16 dtype casting via [`HalfPrecisionAdapter`])
//! - Adapter chaining for composing multiple transformations

use crate::TensorSnapshot;

use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;

use burn_core::tensor::shape;
use burn_core::tensor::{DType, TensorData};
use hashbrown::HashSet;

// Module type names as they appear in the container_type field
// These come from the Module derive macro which uses stringify! on the struct name
// Format: "Struct:TypeName" for user-defined structs
mod module_names {
    // The actual string constants that match what the Module derive macro produces
    pub const LINEAR: &str = "Struct:Linear";
    pub const BATCH_NORM: &str = "Struct:BatchNorm";
    pub const LAYER_NORM: &str = "Struct:LayerNorm";
    pub const GROUP_NORM: &str = "Struct:GroupNorm";
    pub const EMBEDDING: &str = "Struct:Embedding";
    pub const CONV1D: &str = "Struct:Conv1d";
    pub const CONV2D: &str = "Struct:Conv2d";
    pub const CONV3D: &str = "Struct:Conv3d";
    pub const CONV_TRANSPOSE1D: &str = "Struct:ConvTranspose1d";
    pub const CONV_TRANSPOSE2D: &str = "Struct:ConvTranspose2d";
    pub const CONV_TRANSPOSE3D: &str = "Struct:ConvTranspose3d";
    pub const DEFORM_CONV2D: &str = "Struct:DeformConv2d";
    pub const INSTANCE_NORM: &str = "Struct:InstanceNorm";
    pub const RMS_NORM: &str = "Struct:RmsNorm";
    pub const PRELU: &str = "Struct:PRelu";
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

    /// Chain adapters together, applying `self` first and then `next`.
    ///
    /// This is useful when multiple transformations are required when importing model weights
    /// (e.g. PyTorch -> Burn layout conversion, then dtype casting, then custom remapping).
    ///
    /// The semantics follow a simple pipeline:
    /// - `adapt`: `next.adapt(&self.adapt(snapshot))`
    /// - `get_alternative_param_name`: try `self` first; if it returns an alternative name,
    ///   try `next` with that name, otherwise return the first alternative name.
    fn chain<A>(self, next: A) -> ChainAdapter
    where
        Self: Sized + 'static,
        A: ModuleAdapter + 'static,
    {
        ChainAdapter::new(self, next)
    }
}

impl Clone for Box<dyn ModuleAdapter> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Adapter that applies two adapters in sequence.
///
/// This allows composing smaller adapters instead of creating one large monolithic adapter.
#[derive(Clone)]
pub struct ChainAdapter {
    first: Box<dyn ModuleAdapter>,
    second: Box<dyn ModuleAdapter>,
}

impl ChainAdapter {
    /// Create a new adapter chain.
    pub fn new<A, B>(first: A, second: B) -> Self
    where
        A: ModuleAdapter + 'static,
        B: ModuleAdapter + 'static,
    {
        Self {
            first: Box::new(first),
            second: Box::new(second),
        }
    }
}

impl ModuleAdapter for ChainAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let snapshot = self.first.adapt(snapshot);
        self.second.adapt(&snapshot)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        if let Some(name) = self
            .first
            .get_alternative_param_name(param_name, container_type)
        {
            self.second
                .get_alternative_param_name(&name, container_type)
                .or(Some(name))
        } else {
            self.second
                .get_alternative_param_name(param_name, container_type)
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
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

/// Returns the default set of module types that `HalfPrecisionAdapter` converts.
///
/// Includes: Linear, Embedding, all Conv variants, LayerNorm, GroupNorm,
/// InstanceNorm, RmsNorm, PRelu.
///
/// Excludes BatchNorm by default because `running_var` underflows in F16.
fn default_half_precision_modules() -> HashSet<String> {
    let modules = [
        module_names::LINEAR,
        module_names::EMBEDDING,
        module_names::CONV1D,
        module_names::CONV2D,
        module_names::CONV3D,
        module_names::CONV_TRANSPOSE1D,
        module_names::CONV_TRANSPOSE2D,
        module_names::CONV_TRANSPOSE3D,
        module_names::DEFORM_CONV2D,
        module_names::LAYER_NORM,
        module_names::GROUP_NORM,
        module_names::INSTANCE_NORM,
        module_names::RMS_NORM,
        module_names::PRELU,
    ];
    modules.iter().map(|s| s.to_string()).collect()
}

/// Adapter for mixed-precision (F32/F16) model storage.
///
/// Auto-detects conversion direction from the snapshot's dtype:
/// - F32 source -> cast to F16 (typical for saving)
/// - F16 source -> cast to F32 (typical for loading)
/// - Other dtypes -> passed through unchanged
///
/// The same instance works for both `with_to_adapter` (save) and `with_from_adapter` (load).
///
/// By default, converts weights in: Linear, Embedding, Conv*, LayerNorm, GroupNorm,
/// InstanceNorm, RmsNorm, PRelu. BatchNorm is excluded because `running_var` underflows in F16.
///
/// # Examples
///
/// Default usage (same adapter for save and load):
/// ```rust
/// # use burn_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new();
/// // store.with_to_adapter(adapter.clone());  // F32 -> F16 on save
/// // store.with_from_adapter(adapter);        // F16 -> F32 on load
/// ```
///
/// Exclude a module type:
/// ```rust
/// # use burn_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new()
///     .without_module("LayerNorm");
/// ```
///
/// Add a custom module type:
/// ```rust
/// # use burn_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new()
///     .with_module("CustomLayer");
/// ```
#[derive(Debug, Clone)]
pub struct HalfPrecisionAdapter {
    modules: HashSet<String>,
}

impl HalfPrecisionAdapter {
    /// Create a new adapter with the default set of modules.
    pub fn new() -> Self {
        Self {
            modules: default_half_precision_modules(),
        }
    }

    /// Add a module type to convert. Accepts both short (`"MyLayer"`) and
    /// qualified (`"Struct:MyLayer"`) forms.
    ///
    /// Note: short names are mapped to `"Struct:Name"`. If you have an Enum-based
    /// module, use the qualified form `"Enum:MyModule"` explicitly.
    pub fn with_module(mut self, module_type: impl Into<String>) -> Self {
        let name = module_type.into();
        if name.contains(':') {
            self.modules.insert(name);
        } else {
            self.modules.insert(format!("Struct:{}", name));
        }
        self
    }

    /// Remove a module type from conversion. Accepts both short and qualified forms.
    pub fn without_module(mut self, module_type: impl Into<String>) -> Self {
        let name = module_type.into();
        let key = if name.contains(':') {
            name
        } else {
            format!("Struct:{}", name)
        };
        assert!(
            self.modules.contains(&key),
            "without_module called with '{}' which is not in the module set",
            key
        );
        self.modules.remove(&key);
        self
    }

    /// Check whether the tensor belongs to a module that should be converted.
    fn should_convert(&self, snapshot: &TensorSnapshot) -> bool {
        snapshot
            .module_type()
            .is_some_and(|mt| self.modules.contains(&mt))
    }
}

impl Default for HalfPrecisionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleAdapter for HalfPrecisionAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // Determine target dtype from source: F32 -> F16, F16 -> F32, anything else -> skip
        let target_dtype = match snapshot.dtype {
            DType::F32 => DType::F16,
            DType::F16 => DType::F32,
            _ => return snapshot.clone(),
        };

        if !self.should_convert(snapshot) {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();

        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(target_dtype))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            target_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
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
    let transposed_shape = shape![snapshot.shape[1], snapshot.shape[0]];

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
    use alloc::sync::Arc;
    use burn_core::tensor::{DType, Shape, TensorData};
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_module_names_match_burn_nn() {
        // If these types are renamed or moved in `burn-nn`, this test will fail to compile.
        #[allow(unused_imports)]
        use burn_nn::{
            BatchNorm, Embedding, GroupNorm, InstanceNorm, LayerNorm, Linear, PRelu, RmsNorm,
            conv::{
                Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
                DeformConv2d,
            },
        };

        assert_eq!(module_names::LINEAR, "Struct:Linear");
        assert_eq!(module_names::BATCH_NORM, "Struct:BatchNorm");
        assert_eq!(module_names::LAYER_NORM, "Struct:LayerNorm");
        assert_eq!(module_names::GROUP_NORM, "Struct:GroupNorm");
        assert_eq!(module_names::EMBEDDING, "Struct:Embedding");
        assert_eq!(module_names::CONV1D, "Struct:Conv1d");
        assert_eq!(module_names::CONV2D, "Struct:Conv2d");
        assert_eq!(module_names::CONV3D, "Struct:Conv3d");
        assert_eq!(module_names::CONV_TRANSPOSE1D, "Struct:ConvTranspose1d");
        assert_eq!(module_names::CONV_TRANSPOSE2D, "Struct:ConvTranspose2d");
        assert_eq!(module_names::CONV_TRANSPOSE3D, "Struct:ConvTranspose3d");
        assert_eq!(module_names::DEFORM_CONV2D, "Struct:DeformConv2d");
        assert_eq!(module_names::INSTANCE_NORM, "Struct:InstanceNorm");
        assert_eq!(module_names::RMS_NORM, "Struct:RmsNorm");
        assert_eq!(module_names::PRELU, "Struct:PRelu");
    }

    fn create_test_snapshot(path: &str, shape: Shape, container_type: &str) -> TensorSnapshot {
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
        let snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![5, 10]);

        // Linear layer bias should not be transposed
        let snapshot = create_test_snapshot("fc.bias", shape![10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10]);
    }

    #[test]
    fn test_pytorch_to_burn_norm_params() {
        let adapter = PyTorchToBurnAdapter;

        // BatchNorm weight -> gamma
        let snapshot = create_test_snapshot("norm.weight", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.gamma");

        // BatchNorm bias -> beta
        let snapshot = create_test_snapshot("norm.bias", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.beta");
    }

    #[test]
    fn test_burn_to_pytorch_linear_weight() {
        let adapter = BurnToPyTorchAdapter;

        // Linear layer weight should be transposed
        let snapshot = create_test_snapshot("fc.weight", shape![5, 10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10, 5]);
    }

    #[test]
    fn test_burn_to_pytorch_norm_params() {
        let adapter = BurnToPyTorchAdapter;

        // BatchNorm gamma -> weight
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.weight");

        // BatchNorm beta -> bias
        let snapshot = create_test_snapshot("norm.beta", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.bias");
    }

    #[test]
    fn test_transpose_different_dtypes() {
        // Test that transpose works for different data types

        // Test with F32
        let f32_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = transpose_tensor_data(f32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test with I32
        let i32_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [2, 3]);
        let transposed = transpose_tensor_data(i32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.to_vec::<i32>().unwrap();
        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

        // Test with F64
        let f64_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]);
        let transposed = transpose_tensor_data(f64_data);
        assert_eq!(transposed.shape, shape![2, 2]);
        let values = transposed.to_vec::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_no_container_info() {
        let adapter = PyTorchToBurnAdapter;

        // Without container info, adapter returns unchanged for non-norm parameters
        let mut snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        snapshot.container_stack = None;

        // Without container info, no transformation occurs for linear layers
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10, 5]); // No transposition without container info

        // Test a non-linear, non-norm parameter - should pass through unchanged
        let mut snapshot2 = create_test_snapshot("other.weight", shape![10, 5], "Struct:Other");
        snapshot2.container_stack = None;
        let adapted2 = adapter.adapt(&snapshot2);
        assert_eq!(adapted2.shape, shape![10, 5]); // No transposition
    }

    #[derive(Clone)]
    struct RenameParamAdapter {
        from: &'static str,
        to: &'static str,
        called: Arc<AtomicUsize>,
    }

    impl ModuleAdapter for RenameParamAdapter {
        fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
            self.called.fetch_add(1, Ordering::Relaxed);

            let path_stack = match snapshot.path_stack.as_ref() {
                Some(stack) => stack,
                None => return snapshot.clone(),
            };
            let param = match path_stack.last() {
                Some(p) => p.as_str(),
                None => return snapshot.clone(),
            };
            if param != self.from {
                return snapshot.clone();
            }

            let mut new_path = path_stack.to_vec();
            *new_path.last_mut().unwrap() = self.to.to_string();

            TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                new_path,
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            )
        }

        fn get_alternative_param_name(
            &self,
            _param_name: &str,
            _container_type: &str,
        ) -> Option<String> {
            None
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct AltNameAdapter {
        from: &'static str,
        to: &'static str,
        called: Arc<AtomicUsize>,
    }

    impl ModuleAdapter for AltNameAdapter {
        fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
            TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                snapshot.path_stack.clone().unwrap_or_default(),
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            )
        }

        fn get_alternative_param_name(
            &self,
            param_name: &str,
            _container_type: &str,
        ) -> Option<String> {
            self.called.fetch_add(1, Ordering::Relaxed);
            if param_name == self.from {
                Some(self.to.to_string())
            } else {
                None
            }
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_chain_adapter_pipes_adapt() {
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));

        let a = RenameParamAdapter {
            from: "weight",
            to: "a",
            called: called1.clone(),
        };
        let b = RenameParamAdapter {
            from: "a",
            to: "b",
            called: called2.clone(),
        };

        let chain = a.chain(b);
        let snapshot = create_test_snapshot("fc.weight", shape![2, 2], module_names::LINEAR);
        let adapted = chain.adapt(&snapshot);

        assert_eq!(adapted.full_path(), "fc.b");
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_chain_adapter_alternative_name_pipes_and_fallbacks() {
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));

        let a = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "weight",
            to: "scale",
            called: called2.clone(),
        };

        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("scale"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // If the second adapter doesn't have a mapping for the first alternative,
        // fall back to the first alternative name.
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));
        let a = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "something-else",
            to: "unused",
            called: called2.clone(),
        };
        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // If the first adapter doesn't provide an alternative, try the second with the original name.
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));
        let a = AltNameAdapter {
            from: "something-else",
            to: "unused",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called2.clone(),
        };
        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // clone_box must preserve behavior.
        let boxed = chain.clone_box();
        let alt = boxed.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
    }

    #[test]
    fn test_half_precision_f32_to_f16() {
        let adapter = HalfPrecisionAdapter::new();
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);

        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F16);
        assert_eq!(adapted.shape, shape![2, 3]);

        let data = adapted.to_data().unwrap();
        assert_eq!(data.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_f16_to_f32() {
        let adapter = HalfPrecisionAdapter::new();

        // Create an F16 snapshot
        let values = vec![1.0f32; 6];
        let data = TensorData::new(values, shape![2, 3]).convert_dtype(DType::F16);
        let path_parts = vec!["fc".to_string(), "weight".to_string()];
        let snapshot = TensorSnapshot::from_closure(
            Rc::new(move || Ok(data.clone())),
            DType::F16,
            shape![2, 3],
            path_parts,
            vec![module_names::LINEAR.to_string()],
            burn_core::module::ParamId::new(),
        );

        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_batch_norm() {
        let adapter = HalfPrecisionAdapter::new();

        // BatchNorm is excluded by default
        let snapshot = create_test_snapshot("norm.weight", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32); // unchanged
    }

    #[test]
    fn test_half_precision_converts_default_modules() {
        let adapter = HalfPrecisionAdapter::new();

        // Linear
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // Embedding
        let snapshot = create_test_snapshot("emb.weight", shape![100, 64], module_names::EMBEDDING);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // Conv2d
        let snapshot =
            create_test_snapshot("conv.weight", shape![3, 3, 3, 3], module_names::CONV2D);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // LayerNorm (included by default)
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // GroupNorm
        let snapshot = create_test_snapshot("gn.gamma", shape![10], module_names::GROUP_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // RmsNorm
        let snapshot = create_test_snapshot("rms.weight", shape![10], module_names::RMS_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_without_module() {
        let adapter = HalfPrecisionAdapter::new().without_module("LayerNorm");

        // LayerNorm removed from conversion set
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F32);

        // Linear still converted
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_module() {
        let adapter = HalfPrecisionAdapter::new().with_module("CustomLayer");

        // Custom module should now be converted
        let snapshot = create_test_snapshot("custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_qualified_name() {
        let adapter = HalfPrecisionAdapter::new().with_module("Struct:CustomLayer");

        let snapshot = create_test_snapshot("custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_chain() {
        let adapter = PyTorchToBurnAdapter.chain(HalfPrecisionAdapter::new());

        let snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);

        // Should be both transposed and cast
        assert_eq!(adapted.shape, shape![5, 10]);
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_skips_no_container() {
        let adapter = HalfPrecisionAdapter::new();
        let mut snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        snapshot.container_stack = None;

        // No module type info: skip
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_non_float() {
        use burn_core::tensor::quantization::QuantScheme;

        let adapter = HalfPrecisionAdapter::new();

        // QFloat source: skip
        let qfloat_dtype = DType::QFloat(QuantScheme::default());
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        let qfloat_snapshot = TensorSnapshot::from_closure(
            snapshot.clone_data_fn(),
            qfloat_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        );
        let adapted = adapter.adapt(&qfloat_snapshot);
        assert_eq!(adapted.dtype, qfloat_dtype);
    }

    #[test]
    fn test_half_precision_default_module_count() {
        let adapter = HalfPrecisionAdapter::new();
        // 14 modules: Linear, Embedding, Conv1d-3d, ConvTranspose1d-3d,
        // DeformConv2d, LayerNorm, GroupNorm, InstanceNorm, RmsNorm, PRelu
        assert_eq!(adapter.modules.len(), 14);
    }

    #[test]
    fn test_half_precision_without_module_qualified() {
        let adapter = HalfPrecisionAdapter::new().without_module("Struct:LayerNorm");

        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_with_module_batch_norm_opt_in() {
        let adapter = HalfPrecisionAdapter::new().with_module("BatchNorm");

        let snapshot = create_test_snapshot("bn.weight", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }
}
