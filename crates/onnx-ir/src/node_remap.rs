use crate::ArgType;

use super::ir::{AttributeValue, Node, NodeType};

/// Remap node type using kernel shape
pub fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(usize) -> NodeType,
{
    let spatial_dims = match node.attrs.get("kernel_shape") {
        Some(AttributeValue::Int64s(ints)) => ints.len(),
        None => {
            // "kernel_shape" attribute is optional and should be inferred from weights
            // https://onnx.ai/onnx/operators/onnx__Conv.html
            if let ArgType::Tensor(weight) = &node.inputs[1].ty {
                // Weight [out_channels, in_channels, kernel size...]
                weight.rank - 2
            } else {
                panic!("Cannot infer kernel spatial dims");
            }
        }
        _ => panic!("kernel_shape is not an int64s"),
    };
    node.node_type = new_node_type(spatial_dims);
}

/// Remap node type to a more specific one
pub fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            3 => NodeType::Conv3d,
            _ => panic!("Only conv 1d, 2d and 3d are supported"),
        }),
        NodeType::ConvTranspose => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::ConvTranspose1d,
                2 => NodeType::ConvTranspose2d,
                3 => NodeType::ConvTranspose3d,
                _ => panic!("Only conv_transpose 1d, 2d and 3d are supported"),
            })
        }
        NodeType::MaxPool => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::MaxPool1d,
                2 => NodeType::MaxPool2d,
                _ => panic!("Only max_pool 1d and 2d are supported"),
            })
        }
        NodeType::AveragePool => {
            remap_node_with_kernel_shape(node, |spatial_dims| match spatial_dims {
                1 => NodeType::AveragePool1d,
                2 => NodeType::AveragePool2d,
                _ => panic!("Only avg_pool 1d and 2d are supported"),
            })
        }
        _ => (),
    }
}
