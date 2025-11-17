use proc_macro2::TokenStream;
use serde::Serialize;

use burn::record::PrecisionSettings;
use onnx_ir::Argument;

use crate::burn::{BurnImports, Field, Scope};

pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug {
    fn inputs(&self) -> Vec<&Argument>;
    fn outputs(&self) -> Vec<&Argument>;

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [Scope](Scope) struct should be used for [input tensor type](Type::Tensor) access.
    /// The method [use_owned_tensor](Scope::use_owned_tensor) keeps track of tensor reference
    /// count and insert `clone` with necessary.
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

    // TODO Combine field and field_init
    /// (Optional) Declare the type of the field
    ///
    /// # Notes
    ///
    /// This should be implemented when the node has some parameters.
    /// Just one field per type is possible, if the node has multiple types for its parameters, a
    /// tuple can be used.
    ///
    /// Other field functions should be implemented when this one returns something other than None.
    ///   * [field_init](NodeCodegen::field_init) to initialize parameters.
    ///   * [field_serialize](NodeCodegen::field_serialize) to create the model record.
    fn field(&self) -> Option<Field> {
        None
    }

    /// (Optional) Declare how the parameters are initialized.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_init(&self) -> Option<TokenStream> {
        None
    }

    /// (Optional) Declare how the parameters are serialized in a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        panic!("Serialization should be implemented when field_type is not None.");
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
