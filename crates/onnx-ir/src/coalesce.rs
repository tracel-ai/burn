use std::{iter::Peekable, slice::Iter};

use super::{
    from_onnx::GraphData,
    ir::{AttributeValue, Node, NodeType},
    proto_conversion::convert_node_proto,
    protos::NodeProto,
};
use crate::ir::{ArgType, Data, TensorData};

/// The function transforms the graph into a new one where the nodes are coalesced into a single node.
pub fn coalesce(
    node: &mut Node,
    nodes_iter: &mut Peekable<Iter<NodeProto>>,
    graph_data: &mut GraphData,
) {
    #[allow(clippy::single_match)]
    match node.node_type {
        NodeType::Gemm => convert_gemm_to_linear(node, graph_data),
        NodeType::MatMul => {
            convert_matmul_to_linear(node, nodes_iter, graph_data);
        }
        _ => {}
    }
}

/// This function converts a Gemm node into a Linear node
///
/// PyTorch and other frameworks use Gemm node to represent Linear layer.
pub(crate) fn convert_gemm_to_linear(node: &mut Node, graph_data: &mut GraphData) {
    if node.outputs.len() != 1 {
        panic!("Gemm node must have 1 output");
    }
    let straight_linear = match (
        node.attrs.get("alpha"),
        node.attrs.get("beta"),
        node.attrs.get("transB"),
    ) {
        (
            Some(AttributeValue::Float32(alpha)),
            Some(AttributeValue::Float32(beta)),
            Some(AttributeValue::Int64(trans_b)),
        ) => *alpha == 1.0 && *beta == 1.0 && *trans_b == 1,
        _ => false,
    };

    if straight_linear {
        node.node_type = NodeType::Linear;
        node.attrs.remove("alpha");
        node.attrs.remove("beta");
        node.attrs.remove("transB");

        // Transpose the weights
        transpose_linear_node_weights(node, graph_data);
    }
}

// Transpose linear weights (required for Gemm -> Linear conversion)
fn transpose_linear_node_weights(node: &mut Node, graph_data: &mut GraphData) {
    assert!(
        node.inputs.len() > 1,
        "Linear node must have at least 2 input"
    );

    assert!(
        graph_data.has_value(&node.inputs[1].name),
        "Input must have a value"
    );

    // Get the constant node that holds the weight
    let weight_input_name = &node.inputs[1].name;
    let const_node = graph_data
        .get_constant_value(weight_input_name)
        .expect("Weight must be a constant");

    // Get the tensor data from the constant node's attribute
    let tensor_data = match const_node.attrs.get("value") {
        Some(AttributeValue::Tensor(td)) => td.clone(),
        _ => panic!("Constant node must have a tensor value attribute"),
    };

    let data = &tensor_data.data;
    let shape = &tensor_data.shape;

    assert_eq!(shape.len(), 2, "Weight must be a 2D tensor");

    let new_shape = vec![shape[1], shape[0]];

    let new_tensor_data = match data {
        Data::Float32s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float32s(data_t),
                shape: new_shape,
            }
        }
        Data::Float64s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float64s(data_t),
                shape: new_shape,
            }
        }
        Data::Float16s(data) => {
            let data_t = transpose_flattened(data.clone(), shape[0], shape[1]);
            TensorData {
                data: Data::Float16s(data_t),
                shape: new_shape,
            }
        }
        _ => panic!("Only float types are supported for Linear node"),
    };

    // Update the constant node's attribute with the transposed weights
    // We need mutable access to the constant node
    if let Some(constant_node) = graph_data.get_constant_value_mut(weight_input_name) {
        constant_node
            .attrs
            .insert("value".to_string(), AttributeValue::Tensor(new_tensor_data));
    }
}

fn transpose_flattened<T: Copy>(matrix: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(matrix.len(), rows * cols, "Matrix must be flattened");

    let mut transposed: Vec<T> = vec![matrix[0]; matrix.len()];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }

    transposed
}

/// This function converts a MatMul node into a Linear node if possible.
///
/// PyTorch and other frameworks use MatMul node to represent Linear layer.
///
/// This function also converts the following Add node into a Linear node if possible.
/// Add node is used to represent bias in PyTorch.
pub(crate) fn convert_matmul_to_linear(
    node: &mut Node,
    iter_mut: &mut Peekable<Iter<NodeProto>>,
    graph_data: &mut GraphData,
) {
    if node.inputs.len() != 2 {
        panic!("MatMul node must have 2 inputs");
    }

    // if the second input does not have a value, it is not a weight, then proceed to the next node
    if !graph_data.has_value(&node.inputs[1].name) {
        return;
    }

    // Check if the second input is a 2D tensor
    if let ArgType::Tensor(ref tensor_type) = node.inputs[1].ty {
        assert_eq!(tensor_type.rank, 2, "Weight must be a 2D tensor");
    } else {
        panic!("Tensor input is expected");
    }

    // Convert the node to Linear
    node.node_type = NodeType::Linear;

    log::debug!("peeking next node for bias conversion");
    // Check the next node for potential conversion
    if let Some(peek_node) = iter_mut.peek() {
        let peek_node = convert_node_proto(peek_node, graph_data);
        if is_add_node_with_bias(&peek_node, node, graph_data) {
            convert_and_remove_add_node(&peek_node, node);

            // You don't have to remove it if it's never stored in the first place
            let _ = iter_mut.next();
        }
    }
}

/// Helper function to check if the peeked node is an Add node with bias
fn is_add_node_with_bias(peek_node: &Node, current_node: &Node, graph_data: &GraphData) -> bool {
    // Check structural requirements first
    if peek_node.node_type != NodeType::Add || peek_node.inputs.len() != 2 {
        return false;
    }

    // Check if one input is the matmul output and the other has a value
    (peek_node.inputs[0].name == current_node.outputs[0].name
        && graph_data.has_value(&peek_node.inputs[1].name))
        || (peek_node.inputs[1].name == current_node.outputs[0].name
            && graph_data.has_value(&peek_node.inputs[0].name))
}

/// Helper function to convert and remove the Add node
fn convert_and_remove_add_node(bias_node: &Node, current_node: &mut Node) {
    // The bias is whichever input is NOT the matmul output
    let bias_input = if bias_node.inputs[0].name == current_node.outputs[0].name {
        bias_node.inputs[1].clone()
    } else {
        bias_node.inputs[0].clone()
    };

    // Push the bias input and update the output name
    current_node.inputs.push(bias_input);
    current_node.outputs[0]
        .name
        .clone_from(&bias_node.outputs[0].name);
}
