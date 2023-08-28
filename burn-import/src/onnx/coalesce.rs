use std::{iter::Peekable, slice::IterMut};

use super::ir::{AttributeValue, Node, NodeType};
use crate::onnx::ir::{ArgType, Data, TensorType};

/// The function transforms the graph into a new one where the nodes are coalesced into a single node.
pub fn coalesce(nodes: &mut Vec<Node>) {
    let mut iter_mut = nodes.iter_mut().peekable();
    let mut nodes_to_remove: Vec<String> = vec![];
    while let Some(node) = iter_mut.next() {
        match node.node_type {
            NodeType::Gemm => convert_gemm_to_linear(node),
            NodeType::MatMul => {
                convert_matmul_to_linear(node, &mut iter_mut, &mut nodes_to_remove);
            }
            _ => {}
        }
    }

    // Remove nodes instructed by conversation functions
    for node_to_remove in nodes_to_remove {
        nodes.retain(|n| n.name != node_to_remove);
    }
}

/// This function converts a Gemm node into a Linear node
///
/// PyTorch and other frameworks use Gemm node to represent Linear layer.
fn convert_gemm_to_linear(node: &mut Node) {
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
        transpose_linear_node_weights(node);
    } else {
        panic!("Full Gemm node not supported yet.");
    }
}

// Transpose linear weights (required for Gemm -> Linear conversion)
fn transpose_linear_node_weights(node: &mut Node) {
    assert!(
        node.inputs.len() > 1,
        "Linear node must have at least 2 input"
    );

    assert!(node.inputs[1].value.is_some(), "Input must have a value");

    let weight = node.inputs[1]
        .clone()
        .into_tensor()
        .expect("Tensor input is expected");

    assert_eq!(weight.dim, 2, "Weight must be a 2D tensor");

    let shape = weight.shape.unwrap();

    match weight.data.expect("Tensor must have data") {
        Data::Float32s(data) => {
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            node.inputs[1].value = Some(Data::Float32s(data_t));
        }
        Data::Float64s(data) => {
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            node.inputs[1].value = Some(Data::Float64s(data_t));
        }
        Data::Float16s(data) => {
            let data_t = transpose_flattened(data, shape[0], shape[1]);
            node.inputs[1].value = Some(Data::Float16s(data_t));
        }
        _ => panic!("Only float types are supported for Linear node"),
    }
    let shape = Some(vec![shape[1], shape[0]]); // Transpose the shape
    node.inputs[1].ty = ArgType::Tensor(TensorType {
        shape,
        elem_type: weight.elem_type,
        dim: 2,
    });
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
fn convert_matmul_to_linear(
    node: &mut Node,
    iter_mut: &mut Peekable<IterMut<Node>>,
    nodes_to_remove: &mut Vec<String>,
) {
    if node.inputs.len() != 2 {
        panic!("MatMul node must have 2 inputs");
    }

    // Do not convert if the second input does not have a value, and
    // treat it as a normal MatMul node
    if node.inputs[1].value.is_none() {
        return;
    }

    let weight = node.inputs[1]
        .clone()
        .into_tensor()
        .expect("Tensor input is expected");

    assert_eq!(weight.dim, 2, "Weight must be a 2D tensor");

    // Convert the node to Linear
    node.node_type = NodeType::Linear;

    // The following block of code is used to convert the following Add node into this Linear node
    // Add node is used to represent bias in PyTorch.
    let peek_node = iter_mut.peek(); // Peek the next node
    if peek_node.is_some()
        && peek_node.unwrap().node_type == NodeType::Add
        && peek_node.unwrap().inputs.len() == 2

        // Make sure the Add node has a value in one of its inputs and 
        // the other input is the output of this MatMul node
        && (peek_node.unwrap().inputs[0].name == node.outputs[0].name
            && peek_node.unwrap().inputs[1].value.is_some())
            | (peek_node.unwrap().inputs[1].name == node.outputs[0].name
                && peek_node.unwrap().inputs[0].value.is_some())
    {
        // Proceed iteration
        let bias_node = iter_mut.next().unwrap();

        // Copy input value from one of the inputs of the Add node
        if bias_node.inputs[0].value.is_some() {
            node.inputs.push(bias_node.inputs[0].clone());
        } else {
            node.inputs.push(bias_node.inputs[1].clone());
        }

        // Rename the output of MatMul node to the output of Add node
        node.outputs[0].name = bias_node.outputs[0].name.clone();

        // Remove the Add node
        nodes_to_remove.push(bias_node.name.clone());
    };
}
