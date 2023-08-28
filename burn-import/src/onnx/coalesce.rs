use crate::onnx::ir::{ArgType, Data, TensorType};

use super::ir::{AttributeValue, Node, NodeType};

/// The function transforms the graph into a new one where the nodes are coalesced into a single node.
pub fn coalesce(nodes: &mut Vec<Node>) {
    for node in nodes.iter_mut() {
        match node.node_type {
            NodeType::Gemm => convert_gemm(node),
            // TODO Account when linear is converted into MatMul and Add nodes
            _ => {}
        }
    }
}

/// This function converts a Gemm node into a Linear node
///
///  Warning: This function is not complete yet.
///  It only supports the case where the Gemm node is a straight linear transformation.
fn convert_gemm(node: &mut Node) {
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
