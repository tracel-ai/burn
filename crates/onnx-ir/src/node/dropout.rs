//! # Dropout
//!
//! Dropout regularization (identity during inference, random zeroing during training).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Dropout.html>
//!
//! ## Opset Versions
//! - **Opset 1-6**: Dropout with ratio as attribute
//! - **Opset 7-11**: Updated type support
//! - **Opset 12**: Ratio and training_mode moved to inputs; added seed attribute
//! - **Opset 13**: Added optional mask output
//!
//! ## Implementation Notes
//! - Current implementation validates opset 7+ (see FIXME at line 76)
//! - According to spec, operator exists since opset 1
//! - Seed attribute (opset 12+) is mentioned in spec but not currently validated (see TODO at line 111)

use crate::ir::{Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};
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
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::OpsetDependent(vec![
                (1, InputSpec::Exact(1)),     // Opset 1-11: data only
                (12, InputSpec::Range(1, 3)), // Opset 12+: data, ratio (optional), training_mode (optional)
            ]),
            outputs: OutputSpec::Range(1, 2), // 1 or 2 outputs (mask is optional)
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // For opset 12+, ratio is an input (input[1])
        // Only lift it if it's a static constant (has a value)
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Also lift training_mode (input[2]) if it's a static constant
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input count based on opset version - Opset 1-11 has 1 input, Opset 12+ can have up to 3 inputs (data, ratio, training_mode) - Missing opset-specific validation

        // First output: same type as input
        same_as_input(node);

        // Second output (mask): boolean tensor with same shape as input, if present
        if node.outputs.len() == 2 {
            let input_type = &node.inputs[0].ty;
            if let crate::ir::ArgType::Tensor(input_tensor) = input_type {
                node.outputs[1].ty = crate::ir::ArgType::Tensor(crate::ir::TensorType {
                    dtype: crate::ir::DType::Bool,
                    rank: input_tensor.rank,
                    static_shape: input_tensor.static_shape.clone(),
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // TODO: Validate 'seed' attribute mentioned in spec (opset 12+) - currently not handled
        // TODO: Validate ratio value is in range [0.0, 1.0] per ONNX spec - Missing constraint validation - Should return error for invalid ratios
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
            Some(input) => match input.value() {
                None => {
                    // Runtime input - no static value available
                    DropoutInput::Runtime(RuntimeInputRef::new(input.name.clone(), 1))
                }
                Some(tensor_data) => {
                    // Static input - extract the scalar value, converting to f64
                    match tensor_data.scalar_f64() {
                        Ok(prob_value) => DropoutInput::Static(prob_value),
                        Err(_) => {
                            return Err(ProcessError::InvalidAttribute {
                                name: "ratio".to_string(),
                                reason: "must be a float".to_string(),
                            });
                        }
                    }
                }
            },
        };

        let config = DropoutConfig { prob };
        Ok(Some(Box::new(config)))
    }

    fn build_node(&self, builder: NodeBuilder) -> Node {
        let config = builder
            .config
            .expect("Config should be set by extract_config")
            .as_any()
            .downcast_ref::<DropoutConfig>()
            .expect("Wrong config type")
            .clone();

        Node::Dropout {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node_with_attr(ratio: f32) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_float("ratio", ratio)
    }

    fn create_test_node_with_input(ratio: f32) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Dropout, "test_dropout")
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

    fn create_test_node_with_runtime_input() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Dropout, "test_dropout")
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

    // TODO: Add test for mask output - Opset 13+ supports optional mask output (boolean tensor) - Missing test coverage for second output
    // TODO: Add test for training_mode input - Opset 12+ has optional training_mode input (input[2]) - Missing test for this input parameter
    // TODO: Add test for seed attribute - Opset 12+ supports seed attribute for reproducibility - Missing test coverage
    // TODO: Add test for invalid ratio values - Test ratio < 0.0 and ratio > 1.0 should return error per spec - Missing constraint validation test
    // TODO: Add test for ratio=0.0 edge case - Should be identity operation (no dropout) - Missing edge case test
    // TODO: Add test for ratio=1.0 edge case - Should drop all values (output all zeros) - Missing edge case test
    // TODO: Add test for different data types - Spec supports float16, float, double, bfloat16 types - Only testing f32
    // TODO: Add test for opset version transitions - Test attribute vs input behavior for opset 11 vs 12 - Missing opset-specific test
    // TODO: Add test for unexpected attributes - Should validate and reject unknown attributes - Missing attribute validation test
}
