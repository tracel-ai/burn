use crate::util::infer_conv_kernel_shape;

use super::ir::{AttributeValue, Node, NodeType};

/// Remap node type using kernel shape
pub fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(&Vec<i64>) -> NodeType,
{
    if let Some(kernel_shape) = node.attrs.get("kernel_shape") {
        if let AttributeValue::Int64s(ints) = kernel_shape {
            node.node_type = new_node_type(ints);
        } else {
            panic!("kernel_shape is not an int64s");
        }
    } else {
        // Handle conv where "kernel_shape" is optional and can be inferred from weights
        let kernel_shape = infer_conv_kernel_shape(&node.inputs[1].ty);
        node.node_type = new_node_type(&kernel_shape);
    }
}

/// Remap node type to a more specific one
pub fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            3 => NodeType::Conv3d,
            _ => panic!("Only conv 1d, 2d and 3d are supported"),
        }),
        NodeType::ConvTranspose => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::ConvTranspose1d,
            2 => NodeType::ConvTranspose2d,
            3 => NodeType::ConvTranspose3d,
            _ => panic!("Only conv_transpose 1d, 2d and 3d are supported"),
        }),
        NodeType::MaxPool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::MaxPool1d,
            2 => NodeType::MaxPool2d,
            _ => panic!("Only max_pool 1d and 2d are supported"),
        }),
        NodeType::AveragePool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::AveragePool1d,
            2 => NodeType::AveragePool2d,
            _ => panic!("Only avg_pool 1d and 2d are supported"),
        }),
        _ => (),
    }
}
