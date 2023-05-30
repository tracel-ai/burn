use burn::tensor::Tensor;
use burn_ndarray::NdArrayBackend;

use super::ir::{ArgType, AttributeValue, Node, NodeType, TensorData};

type B = NdArrayBackend<f32>;

/// The function transforms the graph into a new one where the nodes are coalesced into a single node.
pub fn coalesce(nodes: &mut Vec<Node>) {
    for node in nodes.iter_mut() {
        match node.node_type {
            NodeType::Gemm => convert_gemm(node),
            _ => {}
        }
    }
}

/// This function converts a Gemm node into a Linear node
///
///  Warning: This function is not complete yet.
///  It only supports the case where the Gemm node is a straight linear transformation.
fn convert_gemm(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Gemm node must have 1 input");
    }

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
    if node.initializers.is_empty() {
        panic!("Linear node must have at least 1 initializer");
    }

    let ArgType::Tensor(node_weight) = node.initializers[0].arg_type.as_ref().unwrap();

    let weight: Tensor<B, 2> = node_weight.try_into().unwrap();

    let weight = weight.transpose();

    let ArgType::Tensor(node_weight) = node.initializers[0].arg_type.as_mut().unwrap();

    node_weight.data = Some(TensorData::Float32(weight.clone().into_data().value));
    node_weight.shape = weight.shape().dims.to_vec();
}
