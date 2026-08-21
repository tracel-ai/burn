//! Module adapters for transforming tensors during save/load
//!
//! This module provides adapters for:
//! - PyTorch/Burn format conversion (weight transposition, parameter renaming)
//! - Mixed-precision storage (F32/F16 dtype casting via [`HalfPrecisionAdapter`])
//! - Adapter chaining for composing multiple transformations

use crate::bridge;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;

use burn_pack::Tensor as PackTensor;

use burn_core::tensor::shape;
use burn_core::tensor::{DType, TensorData};
use hashbrown::HashSet;

// Module type names as they appear in the container stack
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

/// Where in a module hierarchy a tensor was found.
///
/// The one thing a [`PackTensor`] cannot carry. Its contents are Burn module vocabulary
/// (`"Struct:Linear"`, `"Vec"`, `"Enum:ConvType"`), produced by the `Module` derive macro,
/// while `burn-pack` is a format crate with no notion of modules at all. Rather than push that
/// vocabulary into the on-disk type, the traversal hands it to adapters alongside the tensor.
///
/// Borrowed rather than owned: both producers ([`Collector`](crate::Collector) and
/// [`Applier`](crate::Applier)) keep this stack live as they walk the module, and pass it
/// straight through.
#[derive(Debug, Clone, Copy)]
pub struct ModuleContext<'a> {
    container_stack: &'a [String],
}

impl<'a> ModuleContext<'a> {
    /// Wrap the container stack captured at a point in a module traversal.
    ///
    /// The stack runs outermost-first, one entry per level entered, e.g.
    /// `["Struct:Model", "Vec", "Struct:Linear"]`.
    pub fn new(container_stack: &'a [String]) -> Self {
        Self { container_stack }
    }

    /// A context with no module information, for tensors that did not come from a traversal.
    ///
    /// What a tensor read from a file has until it is matched against a module: the format
    /// records names, not the module hierarchy that produced them.
    pub fn none() -> Self {
        Self {
            container_stack: &[],
        }
    }

    /// The innermost user-defined module type, skipping collection wrappers.
    ///
    /// This is what adapters key on, because a parameter's meaning comes from the module that
    /// declares it and not from the `Vec` or array that happens to hold that module.
    ///
    /// # Examples
    /// - `Linear.weight` -> `Some("Struct:Linear")`
    /// - `Vec<Linear>[0].weight` -> `Some("Struct:Linear")`
    /// - `Vec<Param>[0]` (no module) -> `None`
    pub fn module_type(&self) -> Option<&'a str> {
        self.container_stack
            .iter()
            .rev()
            .find(|ct| ct.starts_with("Struct:") || ct.starts_with("Enum:"))
            .map(|s| s.as_str())
    }
}

/// Trait for adapting tensors between different module formats
pub trait ModuleAdapter: Send + Sync {
    /// Adapt a tensor given where in the module hierarchy it was found.
    ///
    /// Takes the tensor by value so a pass-through costs nothing, and returns one whose data is
    /// still deferred: a transform is composed onto the byte source rather than run here.
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor;

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
    /// - `adapt`: `next.adapt(self.adapt(tensor, ctx), ctx)`
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
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        self.second.adapt(self.first.adapt(tensor, ctx), ctx)
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
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        tensor
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Compose a dtype cast onto a tensor's data, leaving it deferred.
fn cast(tensor: PackTensor, target: DType) -> PackTensor {
    let (name, shape) = (tensor.name.clone(), tensor.shape.clone());

    bridge::map_data(&tensor, name, target, shape, move |data| {
        data.convert_dtype(target)
    })
}

/// Replace the last segment of a tensor's name, leaving its data untouched.
fn rename_param(mut tensor: PackTensor, new_name: &str) -> PackTensor {
    let keep = tensor.name.rfind('.').map(|i| i + 1).unwrap_or(0);
    tensor.name.truncate(keep);
    tensor.name.push_str(new_name);
    tensor
}

/// The last segment of a tensor's name, which is the parameter's own name.
fn param_name(tensor: &PackTensor) -> &str {
    tensor.name.rsplit('.').next().unwrap_or("")
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
/// Auto-detects conversion direction from the tensor's dtype:
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
    fn should_convert(&self, ctx: ModuleContext<'_>) -> bool {
        ctx.module_type()
            .is_some_and(|mt| self.modules.contains(mt))
    }
}

impl Default for HalfPrecisionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleAdapter for HalfPrecisionAdapter {
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        // Determine target dtype from source: F32 -> F16, F16 -> F32, anything else -> skip
        let target_dtype = match tensor.dtype {
            DType::F32 => DType::F16,
            DType::F16 => DType::F32,
            _ => return tensor,
        };

        if !self.should_convert(ctx) {
            return tensor;
        }

        cast(tensor, target_dtype)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter that casts every float tensor to one target dtype.
///
/// Unlike [`HalfPrecisionAdapter`], which infers the direction from the
/// source dtype (F32 <-> F16 only) and converts a fixed list of module types,
/// this adapter is *target-driven*: every float tensor (F64, F32, Flex32,
/// F16, BF16) is converted to the given dtype regardless of which module it
/// belongs to. Non-float tensors (ints, bools, quantized) and tensors already
/// in the target dtype pass through unchanged.
///
/// This is the right tool for loading a stock checkpoint into a module whose
/// backend runs at a different float precision, e.g. BF16 safetensors into
/// an F16 model. `HalfPrecisionAdapter` passes BF16 tensors through
/// unconverted and skips custom module types entirely, so the loaded params
/// silently keep the source dtype instead of the module's.
///
/// # Examples
///
/// Cast everything to the backend's float element on load:
/// ```rust,ignore
/// let adapter = FloatCastAdapter::to(<B::FloatElem as Element>::dtype());
/// // store.with_from_adapter(adapter);
/// ```
///
/// Chain after a framework adapter:
/// ```rust
/// # use burn_store::{FloatCastAdapter, ModuleAdapter, PyTorchToBurnAdapter};
/// # use burn_core::tensor::DType;
/// let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F16));
/// ```
#[derive(Debug, Clone)]
pub struct FloatCastAdapter {
    /// The dtype every float tensor is converted to.
    target: DType,
}

impl FloatCastAdapter {
    /// Create an adapter that casts every float tensor to `target`.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not a float dtype.
    pub fn to(target: DType) -> Self {
        assert!(
            target.is_float(),
            "FloatCastAdapter target must be a float dtype, got {target:?}"
        );
        Self { target }
    }
}

impl ModuleAdapter for FloatCastAdapter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        if !tensor.dtype.is_float() || tensor.dtype == self.target {
            return tensor;
        }

        cast(tensor, self.target)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter for converting from PyTorch format to Burn format
///
/// Handles:
/// - Linear layer weight transposition (PyTorch: [out, in] -> Burn: [in, out])
/// - Normalization parameter renaming (weight -> gamma, bias -> beta)
#[derive(Debug, Clone, Default)]
pub struct PyTorchToBurnAdapter;

impl ModuleAdapter for PyTorchToBurnAdapter {
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        adapt_pytorch_tensor(tensor, ctx, PyTorchConversionDirection::PyTorchToBurn)
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
/// - Linear layer weight transposition (Burn: [in, out] -> PyTorch: [out, in])
/// - Normalization parameter renaming (gamma -> weight, beta -> bias)
#[derive(Debug, Clone, Default)]
pub struct BurnToPyTorchAdapter;

impl ModuleAdapter for BurnToPyTorchAdapter {
    fn adapt(&self, tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        adapt_pytorch_tensor(tensor, ctx, PyTorchConversionDirection::BurnToPyTorch)
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
        module_names::BATCH_NORM
            | module_names::LAYER_NORM
            | module_names::GROUP_NORM
            | module_names::RMS_NORM
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
    tensor: PackTensor,
    ctx: ModuleContext<'_>,
    direction: PyTorchConversionDirection,
) -> PackTensor {
    // Get module type for matching (ignores Vec/Array wrappers)
    let Some(module_type) = ctx.module_type() else {
        return tensor; // No user-defined module found
    };
    let param = param_name(&tensor);

    // Decide everything that reads the tensor before moving it, so the borrows end here.
    let transpose =
        module_type == module_names::LINEAR && param == "weight" && tensor.shape.len() == 2;

    // Normalization layers: rename parameters based on direction
    let rename = is_normalization_layer(module_type)
        .then(|| match direction {
            PyTorchConversionDirection::PyTorchToBurn => pytorch_norm_param_to_burn(param),
            PyTorchConversionDirection::BurnToPyTorch => burn_norm_param_to_pytorch(param),
        })
        .flatten();

    // Linear: transpose weight (bidirectional - same operation both ways)
    if transpose {
        return transpose_2d_tensor(tensor);
    }

    if let Some(new_name) = rename {
        return rename_param(tensor, new_name);
    }

    tensor
}

/// Transpose a 2D tensor
fn transpose_2d_tensor(tensor: PackTensor) -> PackTensor {
    if tensor.shape.len() != 2 {
        return tensor;
    }

    let (name, dtype) = (tensor.name.clone(), tensor.dtype);
    let transposed_shape = shape![tensor.shape[1], tensor.shape[0]];

    // Compose the transpose onto the byte source; it runs when the data is finally drawn.
    bridge::map_data(
        &tensor,
        name,
        dtype,
        transposed_shape,
        transpose_tensor_data,
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
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use burn_core::tensor::{Bytes, DType, Shape, TensorData};
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

    /// A tensor holding `data`, standing in for one the collector produced.
    fn tensor_with(name: &str, data: TensorData) -> PackTensor {
        bridge::from_data(data, name.to_string(), None)
    }

    /// An F32 tensor of ones.
    fn tensor(name: &str, shape: Shape) -> PackTensor {
        let values = vec![1.0f32; shape.iter().product()];
        tensor_with(name, TensorData::new(values, shape))
    }

    /// The container stack of a tensor found directly inside `container_type`.
    fn containers(container_type: &str) -> Vec<String> {
        vec![container_type.to_string()]
    }

    /// Adapt an F32 tensor of ones found directly inside `container_type`.
    fn adapt_in(
        adapter: &dyn ModuleAdapter,
        name: &str,
        shape: Shape,
        container_type: &str,
    ) -> PackTensor {
        let containers = containers(container_type);
        adapter.adapt(tensor(name, shape), ModuleContext::new(&containers))
    }

    #[test]
    fn test_pytorch_to_burn_linear_weight() {
        let adapter = PyTorchToBurnAdapter;

        // Linear layer weight should be transposed
        let adapted = adapt_in(&adapter, "fc.weight", shape![10, 5], module_names::LINEAR);
        assert_eq!(adapted.shape, shape![5, 10]);

        // Linear layer bias should not be transposed
        let adapted = adapt_in(&adapter, "fc.bias", shape![10], module_names::LINEAR);
        assert_eq!(adapted.shape, shape![10]);
    }

    #[test]
    fn test_pytorch_to_burn_norm_params() {
        let adapter = PyTorchToBurnAdapter;

        // BatchNorm weight -> gamma
        let adapted = adapt_in(
            &adapter,
            "norm.weight",
            shape![10],
            module_names::BATCH_NORM,
        );
        assert_eq!(adapted.name, "norm.gamma");

        // BatchNorm bias -> beta
        let adapted = adapt_in(&adapter, "norm.bias", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapted.name, "norm.beta");
    }

    #[test]
    fn test_burn_to_pytorch_linear_weight() {
        let adapter = BurnToPyTorchAdapter;

        // Linear layer weight should be transposed
        let adapted = adapt_in(&adapter, "fc.weight", shape![5, 10], module_names::LINEAR);
        assert_eq!(adapted.shape, shape![10, 5]);
    }

    #[test]
    fn test_burn_to_pytorch_norm_params() {
        let adapter = BurnToPyTorchAdapter;

        // BatchNorm gamma -> weight
        let adapted = adapt_in(&adapter, "norm.gamma", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapted.name, "norm.weight");

        // BatchNorm beta -> bias
        let adapted = adapt_in(&adapter, "norm.beta", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapted.name, "norm.bias");
    }

    /// Renaming a parameter must leave the rest of the path alone, including a name with no
    /// path at all (a top-level parameter) and one nested several levels deep.
    #[test]
    fn rename_keeps_the_enclosing_path() {
        let adapter = PyTorchToBurnAdapter;

        let adapted = adapt_in(
            &adapter,
            "encoder.layers.0.norm.weight",
            shape![10],
            module_names::LAYER_NORM,
        );
        assert_eq!(adapted.name, "encoder.layers.0.norm.gamma");

        let adapted = adapt_in(&adapter, "weight", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapted.name, "gamma");
    }

    #[test]
    fn test_transpose_different_dtypes() {
        // Test that transpose works for different data types

        // Test with F32
        let f32_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = transpose_tensor_data(f32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.try_to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test with I32
        let i32_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [2, 3]);
        let transposed = transpose_tensor_data(i32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.try_to_vec::<i32>().unwrap();
        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

        // Test with F64
        let f64_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]);
        let transposed = transpose_tensor_data(f64_data);
        assert_eq!(transposed.shape, shape![2, 2]);
        let values = transposed.try_to_vec::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
    }

    /// A transposed tensor must actually carry transposed bytes, not just a swapped shape.
    /// `map_data` defers the transform, so nothing checks it until the data is drawn.
    #[test]
    fn transpose_moves_the_data_not_just_the_shape() {
        let adapter = PyTorchToBurnAdapter;
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let containers = containers(module_names::LINEAR);

        let adapted = adapter.adapt(
            tensor_with("fc.weight", data),
            ModuleContext::new(&containers),
        );

        assert_eq!(adapted.shape, shape![3, 2]);
        let values = bridge::to_data(&adapted)
            .unwrap()
            .try_into_vec::<f32>()
            .unwrap();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_no_container_info() {
        let adapter = PyTorchToBurnAdapter;

        // Without container info, no transformation occurs for linear layers
        let adapted = adapter.adapt(tensor("fc.weight", shape![10, 5]), ModuleContext::none());
        assert_eq!(adapted.shape, shape![10, 5]); // No transposition without container info

        // Test a non-linear, non-norm parameter - should pass through unchanged
        let adapted = adapter.adapt(tensor("other.weight", shape![10, 5]), ModuleContext::none());
        assert_eq!(adapted.shape, shape![10, 5]); // No transposition
    }

    /// A `Vec<Linear>` still adapts as a Linear: the collection wrapper sits on top of the
    /// module in the container stack, and `module_type` has to look past it.
    #[test]
    fn module_type_looks_past_collection_wrappers() {
        let adapter = PyTorchToBurnAdapter;
        let containers = vec![
            "Struct:Model".to_string(),
            "Vec".to_string(),
            module_names::LINEAR.to_string(),
        ];

        let adapted = adapter.adapt(
            tensor("layers.0.weight", shape![10, 5]),
            ModuleContext::new(&containers),
        );
        assert_eq!(adapted.shape, shape![5, 10]);
    }

    #[derive(Clone)]
    struct RenameParamAdapter {
        from: &'static str,
        to: &'static str,
        called: Arc<AtomicUsize>,
    }

    impl ModuleAdapter for RenameParamAdapter {
        fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
            self.called.fetch_add(1, Ordering::Relaxed);

            if param_name(&tensor) != self.from {
                return tensor;
            }

            rename_param(tensor, self.to)
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
        fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
            tensor
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
        let adapted = adapt_in(&chain, "fc.weight", shape![2, 2], module_names::LINEAR);

        assert_eq!(adapted.name, "fc.b");
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

        let adapted = adapt_in(&adapter, "fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapted.dtype, DType::F16);
        assert_eq!(adapted.shape, shape![2, 3]);

        let data = bridge::to_data(&adapted).unwrap();
        assert_eq!(data.dtype, DType::F16);
    }

    /// The cast has to be reflected in the declared byte length too, not only the dtype: the
    /// writer reserves space from `byte_len` before the provider ever runs, so an F16 tensor
    /// still claiming its F32 length would misplace every tensor written after it.
    #[test]
    fn test_half_precision_updates_byte_len() {
        let adapter = HalfPrecisionAdapter::new();

        let adapted = adapt_in(&adapter, "fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapted.byte_len(), 6 * 2);
        assert_eq!(bridge::to_data(&adapted).unwrap().bytes.len(), 6 * 2);
    }

    #[test]
    fn test_half_precision_f16_to_f32() {
        let adapter = HalfPrecisionAdapter::new();

        // Create an F16 tensor
        let data = TensorData::new(vec![1.0f32; 6], shape![2, 3]).convert_dtype(DType::F16);
        let containers = containers(module_names::LINEAR);

        let adapted = adapter.adapt(
            tensor_with("fc.weight", data),
            ModuleContext::new(&containers),
        );
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_batch_norm() {
        let adapter = HalfPrecisionAdapter::new();

        // BatchNorm is excluded by default
        let adapted = adapt_in(
            &adapter,
            "norm.weight",
            shape![10],
            module_names::BATCH_NORM,
        );
        assert_eq!(adapted.dtype, DType::F32); // unchanged
    }

    #[test]
    fn test_half_precision_converts_default_modules() {
        let adapter = HalfPrecisionAdapter::new();

        for (name, shape, module) in [
            ("fc.weight", shape![2, 3], module_names::LINEAR),
            ("emb.weight", shape![100, 64], module_names::EMBEDDING),
            ("conv.weight", shape![3, 3, 3, 3], module_names::CONV2D),
            ("norm.gamma", shape![10], module_names::LAYER_NORM),
            ("gn.gamma", shape![10], module_names::GROUP_NORM),
            ("rms.weight", shape![10], module_names::RMS_NORM),
        ] {
            assert_eq!(
                adapt_in(&adapter, name, shape, module).dtype,
                DType::F16,
                "{module} should be converted"
            );
        }
    }

    #[test]
    fn test_half_precision_without_module() {
        let adapter = HalfPrecisionAdapter::new().without_module("LayerNorm");

        // LayerNorm removed from conversion set
        let adapted = adapt_in(&adapter, "norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapted.dtype, DType::F32);

        // Linear still converted
        let adapted = adapt_in(&adapter, "fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_module() {
        let adapter = HalfPrecisionAdapter::new().with_module("CustomLayer");

        // Custom module should now be converted
        let adapted = adapt_in(&adapter, "custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_qualified_name() {
        let adapter = HalfPrecisionAdapter::new().with_module("Struct:CustomLayer");

        let adapted = adapt_in(&adapter, "custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_chain() {
        let adapter = PyTorchToBurnAdapter.chain(HalfPrecisionAdapter::new());

        let adapted = adapt_in(&adapter, "fc.weight", shape![10, 5], module_names::LINEAR);

        // Should be both transposed and cast
        assert_eq!(adapted.shape, shape![5, 10]);
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_skips_no_container() {
        let adapter = HalfPrecisionAdapter::new();

        // No module type info: skip
        let adapted = adapter.adapt(tensor("fc.weight", shape![2, 3]), ModuleContext::none());
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_non_float() {
        use burn_core::tensor::quantization::QuantScheme;

        let adapter = HalfPrecisionAdapter::new();

        // QFloat source: skip. Built with the length the scheme actually implies, since every
        // materialization is now checked against the declared byte length.
        let qfloat_dtype = DType::QFloat(QuantScheme::default());
        let shape = shape![2, 3];
        let bytes = Bytes::from_bytes_vec(vec![0u8; bridge::data_len(qfloat_dtype, &shape)]);
        let qfloat = PackTensor::new("fc.weight".to_string(), qfloat_dtype, shape, None, bytes);

        let containers = containers(module_names::LINEAR);
        let adapted = adapter.adapt(qfloat, ModuleContext::new(&containers));
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

        let adapted = adapt_in(&adapter, "norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapted.dtype, DType::F32);
    }

    /// A tensor with a specific float dtype (values convertible exactly in F16/BF16 so
    /// round-trips can be checked precisely).
    fn float_tensor(dtype: DType) -> PackTensor {
        let values = vec![1.0f32, -2.0, 0.5, 4.0, -0.25, 8.0];
        let data = TensorData::new(values, shape![2, 3]).convert_dtype(dtype);
        tensor_with("fc.weight", data)
    }

    #[test]
    fn test_float_cast_bf16_to_f16() {
        // The case HalfPrecisionAdapter cannot handle: BF16 sources.
        let adapter = FloatCastAdapter::to(DType::F16);
        let containers = containers(module_names::LINEAR);

        let adapted = adapter.adapt(float_tensor(DType::BF16), ModuleContext::new(&containers));
        assert_eq!(adapted.dtype, DType::F16);
        assert_eq!(adapted.shape, shape![2, 3]);
        assert_eq!(adapted.name, "fc.weight");

        let data = bridge::to_data(&adapted).unwrap();
        assert_eq!(data.dtype, DType::F16);
        let values = data
            .convert_dtype(DType::F32)
            .try_into_vec::<f32>()
            .unwrap();
        assert_eq!(values, vec![1.0, -2.0, 0.5, 4.0, -0.25, 8.0]);
    }

    #[test]
    fn test_float_cast_converts_any_module_type() {
        // No module-type allowlist: custom modules are converted too.
        let adapter = FloatCastAdapter::to(DType::F16);
        let containers = containers("Struct:CustomLayer");
        let adapted = adapter.adapt(float_tensor(DType::F32), ModuleContext::new(&containers));
        assert_eq!(adapted.dtype, DType::F16);

        // Even with no container info at all.
        let adapted = adapter.adapt(float_tensor(DType::F32), ModuleContext::none());
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_float_cast_passthrough_on_target_dtype() {
        let adapter = FloatCastAdapter::to(DType::F16);
        let adapted = adapter.adapt(float_tensor(DType::F16), ModuleContext::none());
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_float_cast_skips_non_float() {
        let adapter = FloatCastAdapter::to(DType::F16);
        let containers = containers(module_names::EMBEDDING);

        let data = TensorData::new(vec![1i64, 2, 3], shape![3]);
        let adapted = adapter.adapt(tensor_with("idx", data), ModuleContext::new(&containers));
        assert_eq!(adapted.dtype, DType::I64);
    }

    #[test]
    fn test_float_cast_chain_after_pytorch() {
        // Layout conversion then dtype cast, the checkpoint-loading shape.
        let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F16));

        let adapted = adapt_in(&adapter, "fc.weight", shape![10, 5], module_names::LINEAR);
        assert_eq!(adapted.shape, shape![5, 10]);
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    #[should_panic(expected = "must be a float dtype")]
    fn test_float_cast_rejects_non_float_target() {
        let _ = FloatCastAdapter::to(DType::I32);
    }

    #[test]
    fn test_half_precision_with_module_batch_norm_opt_in() {
        let adapter = HalfPrecisionAdapter::new().with_module("BatchNorm");

        let adapted = adapt_in(&adapter, "bn.weight", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapted.dtype, DType::F16);
    }
}
