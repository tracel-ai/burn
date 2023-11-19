use super::ir::{AttributeValue, Node, NodeType};

/// Remap node type using kernel shape
pub fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(&Vec<i64>) -> NodeType,
{
    if let AttributeValue::Int64s(ints) = node.attrs.get("kernel_shape").unwrap() {
        node.node_type = new_node_type(ints);
    } else {
        panic!("kernel_shape is not an int64s");
    }
}

/// Remap node type to a more specific one
pub fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            _ => panic!("Only conv 1d and 2d are supported"),
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
