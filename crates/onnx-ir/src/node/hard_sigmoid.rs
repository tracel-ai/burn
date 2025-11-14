//! # HardSigmoid
//!
//! Applies the HardSigmoid function element-wise to the input tensor, which is a piecewise linear
//! approximation of the sigmoid function. The function clips a linear transformation to the range [0, 1].
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__HardSigmoid.html>
//!
//! ## Formula
//! ```text
//! y = max(0, min(1, alpha * x + beta))
//! ```
//!
//! ## Type Constraints
//! - `T`: float16, float32, float64, bfloat16
//!
//! ## Opset Versions
//! - **Opset 1-5**: Earlier versions with different default values
//! - **Opset 6+**: Current version with alpha=0.2, beta=0.5 as defaults

use crate::ir::{Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for HardSigmoid operation
#[derive(Debug, Clone, Default)]
pub struct HardSigmoidConfig {
    pub alpha: f64,
    pub beta: f64,
}

pub(crate) struct HardSigmoidProcessor;

impl NodeProcessor for HardSigmoidProcessor {
    type Config = HardSigmoidConfig;

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
        // TODO: Validate unexpected attributes before config extraction
        // The spec only supports "alpha" and "beta" attributes
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" | "beta" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for HardSigmoid: {}", key),
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
    ) -> Result<Self::Config, ProcessError> {
        // Extract alpha and beta attributes
        let mut alpha = 0.2;
        let mut beta = 0.5;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => alpha = value.clone().into_f32() as f64,
                "beta" => beta = value.clone().into_f32() as f64,
                _ => {}
            }
        }

        let config = HardSigmoidConfig { alpha, beta };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::HardSigmoid {
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

    fn create_test_node(alpha: f32, beta: f32) -> NodeBuilder {
        TestNodeBuilder::new(NodeType::HardSigmoid, "test_hard_sigmoid")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .attr_float("beta", beta)
            .build()
    }

    #[test]
    fn test_hard_sigmoid_config_with_attrs() {
        let node = create_test_node(0.3, 0.6);
        let mut node = node;
        let processor = HardSigmoidProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!((config.alpha - 0.3).abs() < 1e-6);
        assert!((config.beta - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_config_default() {
        let mut node = create_test_node(0.3, 0.6);
        node.attrs.clear(); // Remove all attributes
        let mut node = node;
        let processor = HardSigmoidProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.alpha, 0.2); // Check default values
        assert_eq!(config.beta, 0.5);
    }
}
