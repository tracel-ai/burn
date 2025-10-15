use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::util::same_as_input;

use crate::ir::{Data, Node, NodeConfig};
use std::any::Any;

/// Represents either a static value or a runtime argument for dropout ratio.
#[derive(Debug, Clone)]
pub enum DropoutInput {
    /// Static ratio known at compile time.
    Static(f64),
    /// Runtime ratio determined during execution.
    Runtime(crate::ir::Argument),
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
        // Validate opset
        if opset < 7 {
            return Err(ProcessError::UnsupportedOpset {
                required: 7,
                actual: opset,
            });
        }

        // Validate we have at least one input
        if node.inputs.is_empty() {
            return Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 0,
            });
        }

        // Validate output count
        if node.outputs.is_empty() {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: 0,
            });
        }

        // Opset 7 and older store probability as an attribute
        if node.attrs.contains_key("ratio") {
            let prob = node.attrs.get("ratio").unwrap().clone().into_f32();
            let config = DropoutConfig {
                prob: DropoutInput::Static(prob as f64),
            };
            node.config = Some(Box::new(config));
            same_as_input(node);
            return Ok(());
        }

        if node.inputs.len() < 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: node.inputs.len(),
            });
        }

        let prob = match node.inputs[1].into_value() {
            None => {
                // Runtime input - no static value available
                let mut runtime_arg = node.inputs[1].clone();
                runtime_arg.value_store = None;
                DropoutInput::Runtime(runtime_arg)
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
        };

        let config = DropoutConfig { prob };
        node.config = Some(Box::new(config));

        // Infer output type
        same_as_input(node);

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
                return Err(ProcessError::MissingInput {
                    index: 1,
                    reason: "Dropout: missing ratio input".to_string(),
                });
            }
            Some(input) => match input.into_value() {
                None => {
                    // Runtime input - no static value available
                    let mut runtime_arg = input.clone();
                    runtime_arg.value_store = None;
                    DropoutInput::Runtime(runtime_arg)
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DropoutConfig>();
        assert!(matches!(&config.prob, DropoutInput::Runtime(arg) if arg.name == "ratio"));
    }

    #[test]
    fn test_dropout_config_missing_input() {
        let mut node = create_test_node_with_input(0.5).build_with_graph_data(16);
        node.attrs.clear(); // Remove attributes
        node.inputs.remove(1); // Remove ratio input
        let mut node = node;
        let processor = DropoutProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }
}
