extern crate alloc;

use alloc::rc::Rc;
use burn_store::{TensorSnapshot, TensorSnapshotError};
use proc_macro2::{Ident, Span, TokenStream};

use onnx_ir::Argument;

use crate::burn::BurnImports;

/// A field in the generated model struct
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Ident,
    pub ty: TokenStream,
    pub init: TokenStream,
}

impl Field {
    pub fn new<S: AsRef<str>>(name: S, ty: TokenStream, init: TokenStream) -> Self {
        if name.as_ref().is_empty() {
            panic!("Field with type {ty:?} was passed with empty name");
        }
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            ty,
            init,
        }
    }
}

/// Tensor kind (Int, Float, Bool)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Int,
    Float,
    Bool,
}

impl From<onnx_ir::ir::DType> for TensorKind {
    fn from(dtype: onnx_ir::ir::DType) -> Self {
        use onnx_ir::ir::DType;

        match dtype {
            DType::F32 => TensorKind::Float,
            DType::F64 => TensorKind::Float,
            DType::I32 => TensorKind::Int,
            DType::I64 => TensorKind::Int,
            DType::I8 | DType::U8 => TensorKind::Int,
            DType::Bool => TensorKind::Bool,
            _ => panic!("Unsupported tensor type"),
        }
    }
}

impl quote::ToTokens for TensorKind {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        // TODO use this throughout the codebase
        let kind = match self {
            TensorKind::Int => quote::quote! { Int },
            TensorKind::Float => quote::quote! { Float },
            TensorKind::Bool => quote::quote! { Bool },
        };
        tokens.extend(kind);
    }
}

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen: std::fmt::Debug {
    /// Returns all input arguments for this node.
    ///
    /// # Notes
    ///
    /// This should return ALL inputs, including static initializers.
    /// Filtering (e.g., for dynamic/constant inputs only) is done at the call site.
    fn inputs(&self) -> &[Argument];

    /// Returns all output arguments for this node.
    ///
    /// # Notes
    ///
    /// This should return ALL outputs.
    fn outputs(&self) -> &[Argument];

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [ScopeAtPosition](super::scope::ScopeAtPosition) encapsulates both the scope and node position.
    /// Use `scope.arg()` to automatically handle Tensor/Scalar/Shape arguments with proper clone tracking.
    fn forward(&self, scope: &mut super::scope::ScopeAtPosition<'_>) -> TokenStream;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

    /// (Optional) Declare the type and initialization of the field
    ///
    /// # Notes
    ///
    /// This should be implemented when the node has some parameters.
    /// Just one field per type is possible, if the node has multiple types for its parameters, a
    /// tuple can be used.
    ///
    /// The returned Field struct contains both the type and initialization code.
    fn field(&self) -> Option<Field> {
        None
    }

    /// (Optional) Collect tensor snapshots for burnpack serialization.
    ///
    /// Returns tensor snapshots with paths like "{field_name}.weight", "{field_name}.bias".
    /// The snapshots must be lazy - data should only be loaded when `to_data()` is called.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name that will be used as the prefix for tensor paths
    ///
    /// # Notes
    ///
    /// For nodes without learnable parameters, the default implementation returns an empty vec.
    fn collect_snapshots(&self, _field_name: &str) -> Vec<TensorSnapshot> {
        vec![]
    }
}

// ============================================================================
// Node utilities
// ============================================================================

/// Helper function to extract tensor data from a node input.
///
/// This is commonly used by nodes that need to access constant tensor values
/// (e.g., weights, biases, normalization parameters).
///
/// # Arguments
///
/// * `inputs` - The node's input arguments
/// * `input_index` - Index of the input to extract data from
///
/// # Returns
///
/// `Some(TensorData)` if the input has a constant value, `None` otherwise
pub fn extract_node_data(
    inputs: &[onnx_ir::Argument],
    input_index: usize,
) -> Option<burn::tensor::TensorData> {
    let input = inputs.get(input_index)?;
    input.value()
}

/// Helper function to convert an Argument's name to a proc_macro2::Ident.
///
/// This is commonly used in the forward() method to generate variable names
/// for inputs and outputs.
///
/// # Arguments
///
/// * `arg` - The argument to convert
///
/// # Returns
///
/// A proc_macro2::Ident with the argument's name
pub fn arg_to_ident(arg: &Argument) -> proc_macro2::Ident {
    proc_macro2::Ident::new(&arg.name, proc_macro2::Span::call_site())
}

// ============================================================================
// Tensor snapshot helpers using NdArray backend
// ============================================================================

/// The backend used for tensor transformations during import.
/// Uses the NdArray backend with f64 for CPU-based tensor operations during ONNX import.
/// We use f64 to preserve maximum precision during intermediate operations,
/// then convert back to the original dtype when creating snapshots.
#[cfg(feature = "onnx")]
pub type SerializationBackend = burn_ndarray::NdArray<f64>;

/// Create a lazy tensor snapshot from an ONNX argument.
///
/// This creates a TensorSnapshot that lazily loads tensor data only when needed.
/// The closure captures the argument and calls `value()` only when `to_data()` is invoked.
///
/// # Arguments
///
/// * `input` - The ONNX argument containing tensor data
/// * `path` - The tensor path (e.g., "linear1.weight")
/// * `container_type` - The container type (e.g., "Linear")
///
/// # Returns
///
/// A TensorSnapshot with lazy data loading
pub fn create_lazy_snapshot(
    input: &Argument,
    path: &str,
    container_type: &str,
) -> Option<TensorSnapshot> {
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use onnx_ir::ir::ArgType;

    // Get tensor metadata without loading data
    let (dtype, shape) = match &input.ty {
        ArgType::Tensor(tensor_type) => {
            let dtype = tensor_type.dtype;
            let shape = tensor_type
                .static_shape
                .as_ref()
                .map(|s| s.to_vec())
                .unwrap_or_default();
            (dtype, shape)
        }
        _ => return None,
    };

    // Clone the input for the closure (lightweight, doesn't copy tensor data)
    let input_clone = input.clone();

    // Create a lazy closure that only loads data when called
    let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
        input_clone.value().ok_or_else(|| {
            TensorSnapshotError::DataError(format!(
                "Failed to extract tensor data for '{}'",
                input_clone.name
            ))
        })
    });

    // Parse path into path_stack
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    Some(TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_stack,
        container_stack,
        ParamId::new(),
    ))
}
