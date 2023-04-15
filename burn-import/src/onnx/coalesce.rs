use super::ir::{AttributeValue, Node, NodeType};

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
        panic!("Gemm node must have 3 inputs");
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
        node.is_stateful = true;
        node.attrs.remove("alpha");
        node.attrs.remove("beta");
        node.attrs.remove("transB");
    } else {
        panic!("Full Gemm node not supported yet.");
    }
}
