use crate::ir::{ArgType, Node, TensorType};

/// Create an EyeLike configuration from the node
/// EyeLike creates an identity matrix with the same shape as the input tensor
pub fn eye_like_config(_node: &Node) {
    // EyeLike operation has no configuration parameters
    // It simply creates an identity matrix with the same shape as the input
}

/// Update output for EyeLike - output has same shape and type as input
pub fn eye_like_update_output(node: &mut Node) {
    log::debug!("EyeLike rank inference for node {}", node.name);

    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            // EyeLike output has the same shape and type as input
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                rank: tensor.rank,
                static_shape: tensor.static_shape.clone(),
            });
            log::debug!("EyeLike output tensor rank: {}", tensor.rank);
        }
        _ => panic!("EyeLike operation requires tensor input"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_eye_like_update_output() {
        let mut node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_f32("output", 2, None) // rank will be updated
            .build();

        eye_like_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![3, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_eye_like_config() {
        let node = NodeBuilder::new(NodeType::EyeLike, "test_eye_like")
            .input_tensor_f32("input", 2, Some(vec![4, 4]))
            .output_tensor_f32("output", 2, None)
            .build();

        // Should not panic - config is trivial for EyeLike
        eye_like_config(&node);
    }
}
