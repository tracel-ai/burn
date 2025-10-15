use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for Flatten operations
#[derive(Debug, Clone)]
pub struct FlattenConfig {
    /// Axis along which to flatten
    pub axis: usize,
}

impl NodeConfig for FlattenConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct FlattenProcessor;

impl NodeProcessor for FlattenProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::util::validate_opset(opset, 9)?;

        // Validate input count
        crate::util::validate_input_count(node, 1)?;

        // Validate output count
        crate::util::validate_output_count(node, 1)?;

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

        // check if the input tensor has at least 2 dimensions
        if tensor.rank < 2 {
            return Err(ProcessError::Custom(format!(
                "Flatten: input tensor must have at least 2 dimensions (got {})",
                tensor.rank
            )));
        }

        // Get reference to config for type inference
        let _config = node.config::<FlattenConfig>();

        // Infer output type - Flatten to a 2D tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 2, ..tensor });

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

        // Extract the axis attribute (default: 1 per ONNX spec)
        let mut axis: i64 = 1;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        let config = FlattenConfig {
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

    fn create_test_node(axis: i64) -> Node {
        NodeBuilder::new(NodeType::Flatten, "test_flatten")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_flatten_config_basic() {
        let node = create_test_node(1);
        let mut node = node;
        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<FlattenConfig>();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_flatten_config_with_negative_axis() {
        let node = create_test_node(-2);
        let mut node = node;
        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<FlattenConfig>();
        assert_eq!(config.axis, 2); // -2 + 4 = 2
    }

    #[test]
    fn test_flatten_config_with_low_rank() {
        let mut node = create_test_node(1);
        // Replace the input with one that has lower rank
        let input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("x", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs[0] = input;
        let mut node = node;
        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_flatten_config_with_multiple_inputs() {
        let mut node = create_test_node(1);
        // Add an extra input
        let extra_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let mut node = node;
        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
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
