use crate::ArgType;

use super::ir::{AttributeValue, Node, NodeType};

/// Remap node type using kernel shape
pub fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(usize) -> NodeType,
{
    let spatial_dims = match node.attrs.get("kernel_shape") {
        Some(AttributeValue::Int64s(ints)) => ints.len(),
        None if [NodeType::Conv, NodeType::ConvTranspose].contains(&node.node_type) => {
            // "kernel_shape" attribute is optional and should be inferred from weights
            // https://onnx.ai/onnx/operators/onnx__Conv.html
            if let ArgType::Tensor(weight) = &node.inputs[1].ty {
                // Skip leading channels in/out
                weight.rank - 2
            } else {
                panic!("Cannot infer kernel spatial dims");
            }
        }
        _ => panic!("Cannot infer kernel shape"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn should_infer_conv2d_node_from_weights_rank() {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [.., k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];

        let mut node = NodeBuilder::new(NodeType::Conv, "test_conv2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data.clone(), weight_shape)
            .output_tensor_f32("output", 4, None)
            // .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", vec![1, 1])
            .attr_ints("pads", vec![0, 0, 0, 0])
            .attr_ints("dilations", vec![1, 1])
            .attr_int("group", 1)
            .build();

        assert_eq!(node.node_type, NodeType::Conv);
        remap_node_type(&mut node);
        assert_eq!(node.node_type, NodeType::Conv2d);
    }

    #[test]
    fn should_infer_conv_transpose1d_node_from_weights_rank() {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [.., kernel_size]
        let weight_shape = vec![2, 2, 4];

        let mut node = NodeBuilder::new(NodeType::ConvTranspose, "test_conv2d")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .output_tensor_f32("output", 3, None)
            .build();

        assert_eq!(node.node_type, NodeType::ConvTranspose);
        remap_node_type(&mut node);
        assert_eq!(node.node_type, NodeType::ConvTranspose1d);
    }
}
