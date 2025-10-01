use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Configuration for NonZero operations
#[derive(Debug, Clone, new)]
pub struct NonZeroConfig {
    // NonZero ONNX operation has no attributes
}

/// Create a NonZero configuration from the node
pub fn nonzero_config(_node: &Node) -> NonZeroConfig {
    // NonZero operation has no configurable attributes
    NonZeroConfig::new()
}

/// Update output for NonZero - output is 2D int64 tensor with shape [input_rank, num_nonzero]
pub fn nonzero_update_output(node: &mut Node) {
    log::debug!("NonZero rank inference for node {}", node.name);

    match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => {
            // Output is always a 2D Int64 tensor
            // Shape: [input_tensor_rank, num_nonzero_elements]
            // First dimension equals input tensor rank
            // Second dimension is dynamic (depends on data)
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 2,
                static_shape: None, // Dynamic shape - second dimension depends on number of nonzero elements
            });
            log::debug!("NonZero output tensor shape: [{}, -1]", tensor.rank);
        }
        _ => panic!("NonZero operation requires tensor input"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_nonzero_update_output() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero")
            .input_tensor_f32("input", 3, Some(vec![2, 3, 4]))
            .output_tensor_i64("output", 2, None) // rank will be updated
            .build();

        nonzero_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_config() {
        let node = NodeBuilder::new(NodeType::NonZero, "test_nonzero")
            .input_tensor_f32("input", 2, Some(vec![3, 3]))
            .output_tensor_i64("output", 2, None)
            .build();

        let config = nonzero_config(&node);
        // NonZero has no attributes, so just verify it constructs successfully
        assert!(matches!(config, NonZeroConfig {}));
    }

    #[test]
    fn test_nonzero_update_output_1d() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero_1d")
            .input_tensor_i32("input", 1, Some(vec![5]))
            .output_tensor_i64("output", 2, None)
            .build();

        nonzero_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_update_output_4d() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero_4d")
            .input_tensor_f64("input", 4, Some(vec![2, 3, 4, 5]))
            .output_tensor_i64("output", 2, None)
            .build();

        nonzero_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
