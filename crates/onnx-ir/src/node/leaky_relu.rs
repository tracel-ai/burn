//! # LeakyRelu
//!
//! Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LeakyRelu.html>
//!
//! ## Type Constraints
//! - `T`: Constrained to float tensors (float16, float32, float64, bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1-5**: Initial version
//! - **Opset 6-15**: Updated with alpha=0.01 as default
//! - **Opset 16+**: Extended type support (added bfloat16)
//!
//! ## Missing Test Coverage
//! - TODO: No test for alpha=0 (should behave like ReLU) - Boundary case not tested
//! - TODO: No test for alpha=1 (identity for negative values) - Boundary case not tested
//! - TODO: No test for negative alpha values - Spec doesn't forbid but behavior unclear
//! - TODO: No test for very large alpha values (e.g., alpha > 1) - Could amplify negative values
//! - TODO: No test for all-positive or all-negative inputs - Edge cases for activation behavior
//! - TODO: No test validating that input must be floating-point type - Integer inputs should be rejected
//! - TODO: No test for zero-size tensors - Empty tensor handling

use crate::ir::{Node, NodeBuilder, NodeConfig};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use std::any::Any;

/// Configuration for LeakyRelu operations
#[derive(Debug, Clone)]
pub struct LeakyReluConfig {
    /// Alpha value for negative slope
    pub alpha: f64,
}

impl NodeConfig for LeakyReluConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct LeakyReluProcessor;

impl NodeProcessor for LeakyReluProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 6,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input tensor dtype is floating-point type - Type constraint T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16) not enforced - burn/crates/onnx-ir/src/node/leaky_relu.rs:55

        // TODO: Validate unexpected attributes before config extraction
        // The spec only supports "alpha" attribute
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LeakyRelu: {}", key),
                    });
                }
            }
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract alpha attribute
        let mut alpha = 0.01;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "alpha" {
                alpha = value.clone().into_f32() as f64
                // TODO: Consider validating alpha >= 0 - Negative alpha values have unclear semantics - burn/crates/onnx-ir/src/node/leaky_relu.rs:88
            }
        }

        let config = LeakyReluConfig { alpha };
        Ok(Some(Box::new(config)))
    }

    fn build_node(&self, builder: NodeBuilder) -> Node {
        let config = builder
            .config
            .expect("Config should be set by extract_config")
            .as_any()
            .downcast_ref::<LeakyReluConfig>()
            .expect("Wrong config type")
            .clone();

        Node::LeakyRelu {
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

    fn create_test_node(alpha: f32) -> NodeBuilder {
        TestNodeBuilder::new(NodeType::LeakyRelu, "test_leaky_relu")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_leaky_relu_config_with_alpha() {
        let node = create_test_node(0.2);
        let mut node = node;
        let processor = LeakyReluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<LeakyReluConfig>();
        assert!((config.alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let mut node = node;
        let processor = LeakyReluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<LeakyReluConfig>();
        assert_eq!(config.alpha, 0.01); // Check default value
    }
}
