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
//! ## Attributes
//! - `alpha` (float, default=0.2): Slope coefficient
//! - `beta` (float, default=0.5): Intercept coefficient
//!
//! ## Inputs
//! - `X` (T): Input tensor of any shape
//!
//! ## Outputs
//! - `Y` (T): Output tensor with the same shape and type as input
//!
//! ## Type Constraints
//! - `T`: float16, float32, float64, bfloat16
//!
//! ## Opset Versions
//! - **Opset 6+**: Current version with alpha and beta attributes

use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use std::any::Any;

/// Configuration for HardSigmoid operation
#[derive(Debug, Clone)]
pub struct HardSigmoidConfig {
    pub alpha: f64,
    pub beta: f64,
}

impl NodeConfig for HardSigmoidConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct HardSigmoidProcessor;

impl NodeProcessor for HardSigmoidProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 6)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

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
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
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
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(alpha: f32, beta: f32) -> Node {
        NodeBuilder::new(NodeType::HardSigmoid, "test_hard_sigmoid")
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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<HardSigmoidConfig>();
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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<HardSigmoidConfig>();
        assert_eq!(config.alpha, 0.2); // Check default values
        assert_eq!(config.beta, 0.5);
    }
}
