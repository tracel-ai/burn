use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};

/// Configuration for ArgMin operations
#[derive(Debug, Clone, new)]
pub struct ArgMinConfig {
    /// Axis along which to find the minimum
    pub axis: usize,
    /// Whether to keep dimensions after reduction
    pub keepdims: bool,
}

/// Create argmin config from the attributes of the node
pub fn argmin_config(node: &Node) -> ArgMinConfig {
    let mut axis: i64 = 0;
    let mut keepdims = true; // default value per ONNX spec

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "ArgMin: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "select_last_index" => {
                if value.clone().into_i64() != 0 {
                    panic!(
                        "select_last_index=1 is not supported for argmin in burn (got {value:?})"
                    );
                }
            }
            "keepdims" => {
                // keepdims=0 and keepdims=1 are both supported
                let keepdims_val = value.clone().into_i64();
                if keepdims_val != 0 && keepdims_val != 1 {
                    panic!(
                        "Only keepdims=0 or keepdims=1 is supported for argmin in burn (got {value:?})",
                    );
                }
                keepdims = keepdims_val != 0;
            }
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    ArgMinConfig::new(axis as usize, keepdims)
}

pub struct ArgMinProcessor;

impl NodeProcessor for ArgMinProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn process(&self, node: &mut Node, _context: &ProcessorContext) {
        log::debug!("ArgMin rank inference for node {}", node.name);

        if node.inputs.len() != 1 {
            panic!("ArgMin: multiple inputs are not supported");
        }
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => panic!("Only tensor input is valid"),
        };

        log::debug!("ArgMin input rank for {}: {}", node.name, tensor.rank);

        // Get config to determine output rank
        let config = argmin_config(node);

        // For burn compatibility, argmin always outputs a tensor
        // When keepdims=false, we still output a tensor but with adjusted rank
        if config.keepdims {
            // keepdims=true: output rank same as input rank (dimension becomes 1)
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: tensor.rank,
                static_shape: None,
            });
        } else if tensor.rank == 1 {
            // keepdims=false on 1D tensor: output is scalar
            node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
        } else {
            // keepdims=false on nD tensor (n > 1): output rank is input rank - 1
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: tensor.rank - 1,
                static_shape: None,
            });
        }

        log::debug!(
            "ArgMin output for {} (keepdims={}): {:?}",
            node.name,
            config.keepdims,
            node.outputs[0].ty
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, select_last_index: i64, keepdims: i64) -> Node {
        NodeBuilder::new(NodeType::ArgMin, "test_argmin")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("output", 3, None)
            .attr_int("axis", axis)
            .attr_int("select_last_index", select_last_index)
            .attr_int("keepdims", keepdims)
            .build()
    }

    #[test]
    fn test_argmin_config_basic() {
        let node = create_test_node(0, 0, 1);
        let config = argmin_config(&node);
        assert_eq!(config.axis, 0);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_argmin_config_negative_axis() {
        let node = create_test_node(-2, 0, 1);
        let config = argmin_config(&node);
        assert_eq!(config.axis, 1); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    #[should_panic(expected = "ArgMin: multiple inputs are not supported")]
    fn test_argmin_config_multiple_inputs() {
        let mut node = create_test_node(0, 0, 1);
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
                static_shape: None,
            }),
            value: None,
        });
        let _ = argmin_config(&node);
    }

    #[test]
    fn test_argmin_config_keepdims_supported() {
        let node_keepdims_0 = create_test_node(0, 0, 0);
        let config_0 = argmin_config(&node_keepdims_0);
        assert_eq!(config_0.axis, 0);
        assert_eq!(config_0.keepdims, false);

        let node_keepdims_1 = create_test_node(0, 0, 1);
        let config_1 = argmin_config(&node_keepdims_1);
        assert_eq!(config_1.axis, 0);
        assert_eq!(config_1.keepdims, true);
    }

    #[test]
    #[should_panic(expected = "Only keepdims=0 or keepdims=1 is supported for argmin in burn")]
    fn test_argmin_config_keepdims_invalid() {
        let node = create_test_node(0, 0, 2); // Invalid keepdims value
        let _ = argmin_config(&node);
    }

    #[test]
    #[should_panic(expected = "select_last_index=1 is not supported for argmin in burn")]
    fn test_argmin_config_select_last_index_invalid() {
        let node = create_test_node(0, 1, 1); // Invalid select_last_index value
        let _ = argmin_config(&node);
    }

    #[test]
    fn test_argmin_update_outputs_keepdims_0() {
        // Test argmin with keepdims=0 - output rank should be reduced but minimum 1 for burn
        let mut node = NodeBuilder::new(NodeType::ArgMin, "test_argmin_keepdims_0")
            .attr_int("axis", 1)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 2, None) // 2D input
            .output_tensor_i64("output", 2, None) // Will be updated by processor
            .build();

        let processor = ArgMinProcessor;
        let context = ProcessorContext::new(16);
        processor.process(&mut node, &context);

        // Should output tensor with rank 1 (2 - 1 = 1, max(1, 1) = 1)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_argmin_update_outputs_keepdims_1() {
        // Test argmin with keepdims=1 - output rank should be same as input
        let mut node = NodeBuilder::new(NodeType::ArgMin, "test_argmin_keepdims_1")
            .attr_int("axis", 0)
            .attr_int("keepdims", 1)
            .input_tensor_f32("data", 3, None) // 3D input
            .output_tensor_i64("output", 3, None) // Will be updated by processor
            .build();

        let processor = ArgMinProcessor;
        let context = ProcessorContext::new(16);
        processor.process(&mut node, &context);

        // Should output tensor with same rank as input (3)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_argmin_update_outputs_keepdims_0_scalar() {
        // Test argmin with keepdims=0 on 1D tensor - should output scalar
        let mut node = NodeBuilder::new(NodeType::ArgMin, "test_argmin_1d_keepdims_0")
            .attr_int("axis", 0)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 1, None) // 1D input
            .output_tensor_i64("output", 1, None) // Will be updated by processor
            .build();

        let processor = ArgMinProcessor;
        let context = ProcessorContext::new(16);
        processor.process(&mut node, &context);

        // Should output scalar (rank 0)
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }
}
