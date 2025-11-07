//! # LayerNormalization
//!
//! Layer normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LayerNormalization.html>
//!
//! ## Attributes
//! - `epsilon` (float, default=1e-5): Numerical stability constant
//! - `axis` (int, default=-1): First normalization dimension
//! - `stash_type` (int, default=1): Computation precision
//!
//! ## Inputs
//! - `X` (T): Input tensor
//! - `Scale` (T): Scale tensor (gamma)
//! - `B` (T, optional): Bias tensor (beta)
//!
//! ## Outputs
//! - `Y` (T): Normalized output tensor
//! - `Mean` (U, optional): Mean values
//! - `InvStdDev` (U, optional): Inverse standard deviation
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version introducing LayerNormalization operator. Supports `axis`,
//!   `epsilon`, and `stash_type` attributes. Includes support for optional Mean and InvStdDev outputs.
//!
//! **Implementation Note**: This implementation validates opset 17+ (MIN constant at line 94).
//! Note that the current implementation requires 3 inputs (including bias) and only produces 1 output,
//! which is more restrictive than the ONNX spec (see FIXMEs at lines 97-101).
//!
//! ## Missing Test Coverage
//! - TODO: No test for optional bias (2 inputs) - Spec allows B to be optional but implementation requires 3 inputs
//! - TODO: No test for custom epsilon values - Only default epsilon=1e-5 tested
//! - TODO: No test for stash_type=0 behavior - Test exists but no verification of computational precision difference
//! - TODO: No test for axis != -1 cases (positive axis values) - Only axis=-1 tested
//! - TODO: No test for edge cases: zero-variance inputs, constant inputs, very large/small values
//! - TODO: No test for optional Mean and InvStdDev outputs - Implementation doesn't support multiple outputs

use crate::ir::{Node, NodeConfig};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for LayerNorm operations
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Number of features/model dimension
    pub d_model: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Whether to use full precision for intermediate calculations (stash_type == 1)
    pub full_precision: bool,
}

impl NodeConfig for LayerNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

impl LayerNormConfig {
    /// Create a new LayerNormConfig
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            epsilon: 1e-5,
            full_precision: true, // Default to true (stash_type default is 1)
        }
    }

    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the full_precision value
    pub fn with_full_precision(mut self, full_precision: bool) -> Self {
        self.full_precision = full_precision;
        self
    }
}

pub struct LayerNormProcessor;

impl NodeProcessor for LayerNormProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::AtLeast(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift scale (input 1) and bias (input 2)
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input tensor dtype is floating-point type - Type constraint T not enforced - burn/crates/onnx-ir/src/node/layer_norm.rs:101
        // TODO: Validate Scale tensor rank matches normalized dimensions - Spec requires Scale to match normalized shape - burn/crates/onnx-ir/src/node/layer_norm.rs:101
        // FIXME: According to ONNX spec, LayerNormalization can have 1-3 outputs
        // (Y is required, Mean and InvStdDev are optional), but we only validate for 1

        // Validate axis attribute before extracting config
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("LayerNorm: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        let mut axis = -1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "epsilon" | "stash_type" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LayerNorm: {key}"),
                    });
                }
            }
        }

        // TODO: Validate epsilon > 0 for numerical stability - Negative or zero epsilon could cause issues - burn/crates/onnx-ir/src/node/layer_norm.rs:132
        // TODO: Validate stash_type is 1 or unspecified - Spec only defines stash_type=1 (float), other values undefined - burn/crates/onnx-ir/src/node/layer_norm.rs:132
        // TODO: Validate axis is within valid range for input tensor rank - Out of bounds axis should be rejected - burn/crates/onnx-ir/src/node/layer_norm.rs:132

        if axis != -1 && axis != weight_shape.len() as i64 - 1 {
            return Err(ProcessError::Custom(
                "LayerNorm: normalization is only supported on the last axis right now".to_string(),
            ));
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
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("LayerNorm: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        let num_features = weight_shape[0];
        let mut epsilon = 1e-5;
        let mut stash_type = 1; // Default value is 1 (full precision)

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => {}
                "epsilon" => epsilon = value.clone().into_f32(),
                "stash_type" => stash_type = value.clone().into_i64(),
                _ => {}
            }
        }

        let full_precision = stash_type == 1;
        let config = LayerNormConfig::new(num_features)
            .with_epsilon(epsilon as f64)
            .with_full_precision(full_precision);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        epsilon: f32,
        axis: i64,
        stash_type: i64,
        num_features: usize,
    ) -> NodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::LayerNormalization, "test_layernorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
            .attr_int("axis", axis)
            .attr_int("stash_type", stash_type)
    }

    #[test]
    fn test_layer_norm_config_basic() {
        let mut node = create_test_node(1e-5, -1, 1, 64).build_with_graph_data(17);
        let processor = LayerNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 17).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        let config = node.config::<LayerNormConfig>();
        assert_eq!(config.d_model, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(config.full_precision); // stash_type == 1
    }

    #[test]
    fn test_layer_norm_config_no_stash_type() {
        let mut node = create_test_node(1e-5, -1, 0, 32).build_with_graph_data(17);
        let processor = LayerNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 17).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        let config = node.config::<LayerNormConfig>();
        assert_eq!(config.d_model, 32);
        assert!(!config.full_precision); // stash_type == 0
    }

    #[test]
    fn test_layer_norm_config_invalid_axis() {
        // For a 1D weight tensor with shape [num_features],
        // both axis=0 (the first and only dim) and axis=-1 (the last dim) are valid
        // So we need to use a 2D weight tensor to test the invalid axis case

        // Create a custom node with a 2D weight tensor
        let weight_data = vec![1.0; 32 * 64]; // 2D weight tensor
        let bias_data = vec![0.0; 32 * 64];

        let node = NodeBuilder::new(NodeType::LayerNormalization, "test_layernorm_invalid")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![32, 64]) // 2D shape
            .input_tensor_f32_data("bias", bias_data, vec![32, 64])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", 1e-5)
            .attr_int("axis", 0) // axis=0 is NOT the last dimension for 2D weight
            .attr_int("stash_type", 1)
            .build_with_graph_data(17);

        // Now axis=0 should trigger an error since it's not the last dimension (1)
        let mut node = node;
        let processor = LayerNormProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
