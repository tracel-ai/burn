use crate::ir::{ArgType, ElementType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use std::any::Any;

/// Configuration for ArgMax operations
#[derive(Debug, Clone, new)]
pub struct ArgMaxConfig {
    /// Axis along which to find the maximum
    pub axis: usize,
    /// Whether to keep dimensions after reduction
    pub keepdims: bool,
}

impl NodeConfig for ArgMaxConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ArgMaxProcessor;

impl NodeProcessor for ArgMaxProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 11)?;
        crate::util::validate_input_count(node, 1)?;
        crate::util::validate_output_count(node, 1)?;

        // Validate select_last_index before config extraction
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "select_last_index" && value.clone().into_i64() != 0 {
                return Err(ProcessError::InvalidAttribute {
                    name: "select_last_index".to_string(),
                    reason: "select_last_index=1 is not supported for argmax in burn".to_string(),
                });
            }
        }

        // Validate keepdims value
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "keepdims" {
                let keepdims_val = value.clone().into_i64();
                if keepdims_val != 0 && keepdims_val != 1 {
                    return Err(ProcessError::InvalidAttribute {
                        name: "keepdims".to_string(),
                        reason: "Only keepdims=0 or keepdims=1 is supported for argmax in burn"
                            .to_string(),
                    });
                }
            }
        }

        // Extract the input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        log::debug!("ArgMax rank inference for node {}", node.name);
        log::debug!("ArgMax input rank for {}: {}", node.name, tensor.rank);

        // Get config values before mutating node
        let keepdims = node.config::<ArgMaxConfig>().keepdims;

        // For burn compatibility, argmax always outputs a tensor
        // When keepdims=false, we still output a tensor but with adjusted rank
        if keepdims {
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
            "ArgMax output for {} (keepdims={}): {:?}",
            node.name,
            keepdims,
            node.outputs[0].ty
        );

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        let mut axis: i64 = 0;
        let mut keepdims = true;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "keepdims" => {
                    let keepdims_val = value.clone().into_i64();
                    keepdims = keepdims_val != 0;
                }
                _ => {}
            }
        }

        if axis < 0 {
            axis += tensor.rank as i64;
        }

        let config = ArgMaxConfig::new(axis as usize, keepdims);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, select_last_index: i64, keepdims: i64) -> Node {
        NodeBuilder::new(NodeType::ArgMax, "test_argmax")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("output", 3, None)
            .attr_int("axis", axis)
            .attr_int("select_last_index", select_last_index)
            .attr_int("keepdims", keepdims)
            .build()
    }

    #[test]
    fn test_argmax_config_basic() {
        let mut node = create_test_node(0, 0, 1);

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ArgMaxConfig>();
        assert_eq!(config.axis, 0);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_argmax_config_negative_axis() {
        let mut node = create_test_node(-2, 0, 1);

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ArgMaxConfig>();
        assert_eq!(config.axis, 1); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_argmax_config_multiple_inputs() {
        let mut node = create_test_node(0, 0, 1);
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });

        let processor = ArgMaxProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_argmax_config_keepdims_supported() {
        let mut node_keepdims_0 = create_test_node(0, 0, 0);

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node_keepdims_0, 16).unwrap();
        node_keepdims_0.config = config;

        let prefs = OutputPreferences::new();
        processor
            .infer_types(&mut node_keepdims_0, 16, &prefs)
            .unwrap();

        let config_0 = node_keepdims_0.config::<ArgMaxConfig>();
        assert_eq!(config_0.axis, 0);
        assert_eq!(config_0.keepdims, false);

        let mut node_keepdims_1 = create_test_node(0, 0, 1);

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node_keepdims_1, 16).unwrap();
        node_keepdims_1.config = config;

        let prefs = OutputPreferences::new();
        processor
            .infer_types(&mut node_keepdims_1, 16, &prefs)
            .unwrap();

        let config_1 = node_keepdims_1.config::<ArgMaxConfig>();
        assert_eq!(config_1.axis, 0);
        assert_eq!(config_1.keepdims, true);
    }

    #[test]
    fn test_argmax_config_keepdims_invalid() {
        let mut node = create_test_node(0, 0, 2); // Invalid keepdims value

        let processor = ArgMaxProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_argmax_config_select_last_index_invalid() {
        let mut node = create_test_node(0, 1, 1); // Invalid select_last_index value

        let processor = ArgMaxProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_argmax_update_outputs_keepdims_0() {
        // Test argmax with keepdims=0 - output rank should be reduced but minimum 1 for burn
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_keepdims_0")
            .attr_int("axis", 1)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 2, None) // 2D input
            .output_tensor_i64("output", 2, None) // Will be updated by processor
            .build();

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
    fn test_argmax_update_outputs_keepdims_1() {
        // Test argmax with keepdims=1 - output rank should be same as input
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_keepdims_1")
            .attr_int("axis", 0)
            .attr_int("keepdims", 1)
            .input_tensor_f32("data", 3, None) // 3D input
            .output_tensor_i64("output", 3, None) // Will be updated by processor
            .build();

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
    fn test_argmax_update_outputs_keepdims_0_scalar() {
        // Test argmax with keepdims=0 on 1D tensor - should output scalar
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_1d_keepdims_0")
            .attr_int("axis", 0)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 1, None) // 1D input
            .output_tensor_i64("output", 1, None) // Will be updated by processor
            .build();

        let processor = ArgMaxProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should output scalar (rank 0)
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected scalar output, got {:?}", other),
        }
    }
}
