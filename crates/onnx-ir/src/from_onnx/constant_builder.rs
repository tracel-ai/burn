//! Constant node creation from ONNX initializers
//!
//! This module handles the conversion of ONNX initializers (weights, biases)
//! into Constant nodes with Static inputs for the IR graph.

use std::collections::HashMap;

use crate::ir::{ArgType, Argument, Node, NodeType, TensorId};
use crate::protos::TensorProto;

use super::tensor_store::TensorStore;

/// Result of processing initializers
pub(super) struct ProcessedConstants {
    /// Created Constant nodes
    pub nodes: Vec<Node>,
    /// Map of constant output names to node indices
    pub constant_nodes: HashMap<String, usize>,
}

/// Create a Constant node with a Static input
///
/// The constant's value is stored in the tensor store and referenced via data_id.
/// The input has an empty name and ValueSource::Static.
/// The output has a named output and ValueSource::Constant.
pub(super) fn create_constant_node(
    node_name: String,
    output_name: String,
    ty: ArgType,
    data_id: TensorId,
) -> Node {
    Node {
        node_type: NodeType::Constant,
        name: node_name,
        inputs: vec![Argument {
            name: String::new(),
            ty: ty.clone(),
            data_id: Some(data_id),
            value_source: crate::ir::ValueSource::Static,
            value_store: None,
        }],
        outputs: vec![Argument {
            name: output_name,
            ty,
            data_id: None,
            value_source: crate::ir::ValueSource::Constant,
            value_store: None,
        }],
        attrs: HashMap::new(),
        config: None,
    }
}

/// Process ONNX initializers into Constant nodes
///
/// Converts all initializers (weights, biases, etc.) into Constant nodes,
/// stores their tensor data in the tensor store, and returns tracking maps.
pub(super) fn process_initializers(
    initializers: &[TensorProto],
    tensor_store: &mut TensorStore,
) -> ProcessedConstants {
    let mut nodes = Vec::new();
    let mut constant_nodes = HashMap::new();

    for initializer in initializers.iter() {
        let (_arg, data) = Argument::from_initializer(initializer);

        // Allocate ID and store tensor data
        let data_id = tensor_store.store(data);

        let idx = nodes.len();
        let const_name = format!("constant{}", idx + 1);
        let output_name = format!("{}_out1", const_name);

        let constant_node =
            create_constant_node(const_name, output_name.clone(), _arg.ty.clone(), data_id);

        constant_nodes.insert(output_name, idx);
        nodes.push(constant_node);
    }

    ProcessedConstants {
        nodes,
        constant_nodes,
    }
}

#[cfg(test)]
/// Create a test constant node with tensor data
pub(super) fn create_test_constant(
    name: String,
    data: crate::ir::Data,
    shape: Vec<usize>,
    tensor_store: &mut TensorStore,
) -> (Node, usize) {
    use crate::ir::TensorData;

    let elem_type = match &data {
        crate::ir::Data::Bool(_) | crate::ir::Data::Bools(_) => crate::ElementType::Bool,
        crate::ir::Data::Float16(_) | crate::ir::Data::Float16s(_) => crate::ElementType::Float16,
        crate::ir::Data::Float32(_) | crate::ir::Data::Float32s(_) => crate::ElementType::Float32,
        crate::ir::Data::Float64(_) | crate::ir::Data::Float64s(_) => crate::ElementType::Float64,
        crate::ir::Data::Uint16(_) | crate::ir::Data::Uint16s(_) => crate::ElementType::Uint16,
        crate::ir::Data::Uint8(_) | crate::ir::Data::Uint8s(_) => crate::ElementType::Uint8,
        crate::ir::Data::Int8(_) | crate::ir::Data::Int8s(_) => crate::ElementType::Int8,
        crate::ir::Data::Int32(_) | crate::ir::Data::Int32s(_) => crate::ElementType::Int32,
        crate::ir::Data::Int64(_) | crate::ir::Data::Int64s(_) => crate::ElementType::Int64,
        crate::ir::Data::String(_) | crate::ir::Data::Strings(_) => crate::ElementType::String,
    };

    let ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
        elem_type,
        rank: shape.len(),
        static_shape: Some(shape.clone()),
    });

    let data_id = tensor_store.store(TensorData { data, shape });

    let output_name = format!("{}_const_out", name);
    let const_node_name = format!("{}_const", name);
    let constant_node = create_constant_node(const_node_name, output_name, ty, data_id);

    // Return node and a placeholder index (caller will assign proper index)
    (constant_node, 0)
}
