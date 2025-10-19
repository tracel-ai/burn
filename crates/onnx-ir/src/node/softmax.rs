use crate::ir::{ArgType, Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for Softmax operations
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Axis along which to apply softmax
    pub axis: usize,
}

impl NodeConfig for SoftmaxConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct SoftmaxProcessor;

impl NodeProcessor for SoftmaxProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 13)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Extract the axis attribute (default: -1 per ONNX spec)
        let mut axis: i64 = -1;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        let config = SoftmaxConfig {
            axis: axis as usize,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Softmax, "test_softmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_softmax_config_basic() {
        let node = create_test_node(-1, 3);
        let mut node = node;
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SoftmaxConfig>();
        assert_eq!(config.axis, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_softmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let mut node = node;
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SoftmaxConfig>();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_softmax_config_multiple_inputs() {
        let mut node = create_test_node(1, 3);
        // Add an extra input
        let extra_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let mut node = node;
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }
}
