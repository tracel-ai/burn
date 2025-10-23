//! # BatchNormalization
//!
//! Batch normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BatchNormalization.html>
//!
//! ## Attributes
//! - `epsilon` (float, default=1e-5): Numerical stability constant
//! - `momentum` (float, default=0.9): Momentum for running statistics
//! - `training_mode` (int, default=0): Training mode flag
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x ...)
//! - `scale` (T): Scale tensor (C)
//! - `B` (T): Bias tensor (C)
//! - `input_mean` (T): Mean tensor (C)
//! - `input_var` (T): Variance tensor (C)
//!
//! ## Outputs
//! - `Y` (T): Normalized output tensor
//! - `mean` (T, optional): Running mean (training mode)
//! - `var` (T, optional): Running variance (training mode)
//! - `saved_mean` (T, optional): Saved mean (training mode)
//! - `saved_var` (T, optional): Saved variance (training mode)
//!
//! ## Opset Versions
//! - **Opset 1-5**: Initial version with spatial attribute
//! - **Opset 6-8**: Removed spatial attribute, added consumed_inputs
//! - **Opset 9-13**: Removed consumed_inputs attribute
//! - **Opset 14-15**: Added training_mode attribute, expanded type support
//! - **Opset 15+**: Current version with full training mode support

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for BatchNorm operations
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
}

impl BatchNormConfig {
    /// Create a new BatchNormConfig
    pub fn new(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        Self {
            num_features,
            epsilon,
            momentum,
        }
    }
}

impl NodeConfig for BatchNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct BatchNormProcessor;

impl NodeProcessor for BatchNormProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift scale (input[1]), bias (input[2]), mean (input[3]), and variance (input[4])
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }
        if node.inputs.len() > 4 && node.inputs[4].is_constant() {
            node.inputs[4].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 9)?;

        // Validate input count (X, scale, B, mean, var) - exactly 5 inputs required
        crate::processor::validate_input_count(node, 5)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Add validation for unexpected attributes
        // TODO: Check training_mode attribute - spec mentions it but implementation doesn't validate it
        // According to spec, training mode outputs mean/var/saved_mean/saved_var which are not currently handled

        // Extract input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // BatchNorm preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let weight_tensor = node.inputs[1].value().ok_or_else(|| {
            ProcessError::Custom("BatchNorm: weight tensor must be present".to_string())
        })?;

        let weight_shape = weight_tensor.shape;
        let num_features = weight_shape[0];

        let mut epsilon = 0f32;
        let mut momentum = 0f32;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "momentum" => momentum = value.clone().into_f32(),
                "epsilon" => epsilon = value.clone().into_f32(),
                _ => {}
            }
        }

        let config = BatchNormConfig::new(num_features, epsilon as f64, momentum as f64);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(epsilon: f32, momentum: f32, num_features: usize) -> NodeBuilder {
        let ones = vec![1.0; num_features];
        let zeros = vec![0.0; num_features];

        NodeBuilder::new(NodeType::BatchNormalization, "test_batchnorm")
            .input_tensor_f32("X", 4, None) // NCHW format
            .input_tensor_f32_data("scale", ones.clone(), vec![num_features])
            .input_tensor_f32_data("bias", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("mean", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("var", ones.clone(), vec![num_features])
            .output_tensor_f32("output", 4, None)
            .attr_float("epsilon", epsilon)
            .attr_float("momentum", momentum)
    }

    #[test]
    fn test_batch_norm_config_basic() {
        let node = create_test_node(1e-5, 0.9, 64).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BatchNormConfig>();

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(f64::abs(config.momentum - 0.9) < 1e-6);
    }

    #[test]
    fn test_batch_norm_config_default_values() {
        let node = create_test_node(0.0, 0.0, 32).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BatchNormConfig>();

        assert_eq!(config.num_features, 32);
        assert!(f64::abs(config.epsilon - 0.0) < 1e-6);
        assert!(f64::abs(config.momentum - 0.0) < 1e-6);
    }
}
