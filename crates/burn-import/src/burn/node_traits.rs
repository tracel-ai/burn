use proc_macro2::{Ident, Span, TokenStream};

use burn::record::PrecisionSettings;
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

pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug {
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
    ///
    /// Other field functions should be implemented when this one returns something other than None.
    ///   * [field_serialize](NodeCodegen::field_serialize) to create the model record.
    fn field(&self) -> Option<Field> {
        None
    }

    /// (Optional) Declare how the parameters are serialized in a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    /// For nodes with fields but no learned parameters (e.g., pooling, dropout),
    /// the default implementation serializes unit.
    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::Serialize;
        ().serialize(serializer)
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
