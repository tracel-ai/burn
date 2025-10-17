use crate::ir::{
    ArgType, Data, ElementType, Node, NodeConfig, RuntimeInputRef, TensorData, TensorType,
};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for the Range operation.
#[derive(Debug, Clone)]
pub struct RangeConfig {
    pub start: RangeInput,
    pub limit: RangeInput,
    pub delta: RangeInput,
}

impl NodeConfig for RangeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for range parameters.
#[derive(Debug, Clone)]
pub enum RangeInput {
    /// Static value known at compile time.
    Static(i64),
    /// Runtime argument determined during execution .
    Runtime(RuntimeInputRef),
}

pub struct RangeProcessor;

impl NodeProcessor for RangeProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        let mut lifted = Vec::new();

        // Lift start input (input[0])
        if !node.inputs.is_empty() {
            lifted.push(node.inputs[0].name.clone());
        }

        // Lift limit input (input[1])
        if node.inputs.len() > 1 {
            lifted.push(node.inputs[1].name.clone());
        }

        // Lift delta input (input[2])
        if node.inputs.len() > 2 {
            lifted.push(node.inputs[2].name.clone());
        }

        Ok(lifted)
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Opset validation
        crate::util::validate_opset(opset, 11)?;

        // Validate input count
        crate::util::validate_input_count(node, 3)?;

        log::debug!("Range rank inference for node {}", node.name);
        log::debug!(
            "Range operation always produces rank 1 tensor for {}",
            node.name
        );

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: None,
        });

        log::debug!("Range output rank for {}: 1", node.name);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Helper function to extract range input
        fn get_range_input(
            node: &Node,
            index: usize,
            param_name: &str,
        ) -> Result<RangeInput, ProcessError> {
            let input = node.inputs.get(index).ok_or_else(|| {
                ProcessError::MissingInput(format!("Range: {} parameter is required", param_name))
            })?;

            match input.into_value() {
                None => Ok(RangeInput::Runtime(RuntimeInputRef::new(
                    input.name.clone(),
                    index,
                ))),
                Some(TensorData {
                    data: Data::Int64s(values),
                    ..
                }) if values.len() == 1 => Ok(RangeInput::Static(values[0])),
                Some(TensorData {
                    data: Data::Int32s(values),
                    ..
                }) if values.len() == 1 => Ok(RangeInput::Static(values[0] as i64)),
                Some(_) => Err(ProcessError::TypeMismatch {
                    expected: "scalar int value".to_string(),
                    actual: format!("{} must be a scalar int value", param_name),
                }),
            }
        }

        let start = get_range_input(node, 0, "start")?;
        let limit = get_range_input(node, 1, "limit")?;
        let delta = get_range_input(node, 2, "delta")?;

        let config = RangeConfig {
            start,
            limit,
            delta,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node() -> Node {
        NodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_i64("start")
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None) // Rank 0 will be updated
            .build()
    }

    #[test]
    fn test_range_output() {
        let mut node = create_test_node();
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_range_missing_inputs() {
        let mut node = create_test_node();
        node.inputs.pop();
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 3,
                actual: 2
            })
        ));
    }
}
