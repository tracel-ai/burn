use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::util::same_as_input;

use crate::ir::{RuntimeInputRef, Data, Node, NodeConfig};
use std::any::Any;

/// Represents either a static value or a runtime argument for dropout ratio.
#[derive(Debug, Clone)]
pub enum DropoutInput {
    /// Static ratio known at compile time.
    Static(f64),
    /// Runtime ratio determined during execution.
    Runtime(RuntimeInputRef),
}

/// Configuration for Dropout operations
#[derive(Debug, Clone)]
pub struct DropoutConfig {
    /// Probability of dropping out a unit
    pub prob: DropoutInput,
}

impl NodeConfig for DropoutConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct DropoutProcessor;

impl NodeProcessor for DropoutProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 7)?;
        crate::util::validate_min_inputs(node, 1)?;

        // Dropout can have 1 or 2 outputs (second output is optional mask)
        if node.outputs.is_empty() || node.outputs.len() > 2 {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        // First output: same type as input
        same_as_input(node);

        // Second output (mask): boolean tensor with same shape as input, if present
        if node.outputs.len() == 2 {
            let input_type = &node.inputs[0].ty;
            if let crate::ir::ArgType::Tensor(input_tensor) = input_type {
                node.outputs[1].ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
                    elem_type: crate::ir::ElementType::Bool,
                    rank: input_tensor.rank,
                    static_shape: input_tensor.static_shape.clone(),
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Opset 7 and older store probability as an attribute
        if node.attrs.contains_key("ratio") {
            let prob = node.attrs.get("ratio").unwrap().clone().into_f32();
            let config = DropoutConfig {
                prob: DropoutInput::Static(prob as f64),
            };
            return Ok(Some(Box::new(config)));
        }

        // Opset 12+ uses input for ratio
        let prob = match node.inputs.get(1) {
            None => {
                return Err(ProcessError::MissingInput(
                    "Dropout: missing ratio input".to_string(),
                ));
            }
            Some(input) => match input.into_value() {
                None => {
                    // Runtime input - no static value available
                    DropoutInput::Runtime(RuntimeInputRef::new(input.name.clone(), 1))
                }
                Some(tensor_data) => {
                    let ratio = tensor_data.data.into_scalar();
                    let prob_value = match ratio {
                        Data::Float16(ratio) => f64::from(f32::from(ratio)),
                        Data::Float32(ratio) => ratio as f64,
                        Data::Float64(ratio) => ratio,
                        _ => {
                            return Err(ProcessError::InvalidAttribute {
                                name: "ratio".to_string(),
                                reason: "must be a float".to_string(),
                            });
                        }
                    };
                    DropoutInput::Static(prob_value)
                }
            },
        };

        let config = DropoutConfig { prob };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node_with_attr(ratio: f32) -> NodeBuilder {
        NodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_float("ratio", ratio)
    }

    fn create_test_node_with_input(ratio: f32) -> NodeBuilder {
        NodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .input_scalar_tensor_f32("ratio", Some(ratio))
            .output_tensor_f32("output", 3, None)
    }

    #[test]
    fn test_dropout_config_with_attr() {
        let node = create_test_node_with_attr(0.3).build_with_graph_data(16);
        let mut node = node;
        let processor = DropoutProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DropoutConfig>();
        assert!(matches!(&config.prob, DropoutInput::Static(v) if f64::abs(*v - 0.3) < 1e-6));
    }

    #[test]
    fn test_dropout_config_with_input() {
        let node = create_test_node_with_input(0.5).build_with_graph_data(16);
        let mut node = node;
        let processor = DropoutProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DropoutConfig>();
        assert!(matches!(&config.prob, DropoutInput::Static(v) if f64::abs(*v - 0.5) < 1e-6));
    }

    fn create_test_node_with_runtime_input() -> NodeBuilder {
        NodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32("ratio", 0, None) // Runtime input - no static value
            .output_tensor_f32("output", 3, None)
    }

    #[test]
    fn test_dropout_config_with_runtime_input() {
        let node = create_test_node_with_runtime_input().build();
        let mut node = node;
        let processor = DropoutProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DropoutConfig>();
        assert!(matches!(&config.prob, DropoutInput::Runtime(arg) if arg.name == "ratio"));
    }

    #[test]
    fn test_dropout_config_missing_input() {
        let mut node = create_test_node_with_input(0.5).build_with_graph_data(16);
        node.attrs.clear(); // Remove attributes
        node.inputs.remove(1); // Remove ratio input
        let node = node;
        let processor = DropoutProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingInput(_))));
    }
}
