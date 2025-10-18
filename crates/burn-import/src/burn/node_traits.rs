use proc_macro2::TokenStream;
use serde::Serialize;

// Import the generated Node enum and match_all! macro from registry
use super::node_registry::{Node, match_all};

use burn::record::PrecisionSettings;

use crate::burn::{BurnImports, Scope, Type};

pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug {
    /// All types that are used as inputs during the forward pass.
    ///
    /// # Notes
    /// The vec should not include types that are accessible with `self`.
    /// See [field type](NodeCodegen::field_type).
    fn input_types(&self) -> Vec<Type>;

    /// All types that are produced during the forward pass.
    fn output_types(&self) -> Vec<Type>;

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [Scope](Scope) struct should be used for [input tensor type](Type::Tensor) access.
    /// The method [use_owned_tensor](Scope::use_owned_tensor) keeps track of tensor reference
    /// count and insert `clone` with necessary.
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    /// Convert the node implementation into a [node entry](Node).
    fn into_node(self) -> Node<PS>;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

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
    fn field_type(&self) -> Option<Type> {
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

impl<PS: PrecisionSettings> Serialize for Node<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.field_serialize(serializer)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for Node<PS> {
    fn output_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::output_types)
    }

    fn input_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::input_types)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match_all!(self, |node| NodeCodegen::<PS>::forward(
            node,
            scope,
            node_position
        ))
    }

    fn field_type(&self) -> Option<Type> {
        match_all!(self, NodeCodegen::<PS>::field_type)
    }

    fn field_init(&self) -> Option<TokenStream> {
        match_all!(self, |node| NodeCodegen::<PS>::field_init(node,))
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match_all!(self, |node| NodeCodegen::<PS>::register_imports(
            node, imports
        ))
    }

    fn into_node(self) -> Node<PS> {
        self
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match_all!(self, |node| NodeCodegen::<PS>::field_serialize(
            node, serializer
        ))
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
/// * `node` - The ONNX IR node
/// * `input_index` - Index of the input to extract data from
///
/// # Returns
///
/// `Some(TensorData)` if the input has a constant value, `None` otherwise
pub fn extract_node_data<E: burn::tensor::Element>(
    node: &onnx_ir::Node,
    input_index: usize,
) -> Option<burn::tensor::TensorData> {
    use burn::tensor::TensorData;

    let input = node.inputs.get(input_index)?;
    let value = input.value()?;

    use onnx_ir::ir::Data;
    let data = match &value.data {
        Data::Float16s(val) => TensorData::new(val.clone(), value.shape.clone()).convert::<E>(),
        Data::Float32s(val) => TensorData::new(val.clone(), value.shape.clone()).convert::<E>(),
        Data::Float64s(val) => TensorData::new(val.clone(), value.shape.clone()).convert::<E>(),
        Data::Int32s(val) => TensorData::new(val.clone(), value.shape.clone()).convert::<E>(),
        Data::Int64s(val) => TensorData::new(val.clone(), value.shape.clone()).convert::<E>(),
        _ => panic!("Unsupported tensor element type"),
    };

    Some(data)
}
