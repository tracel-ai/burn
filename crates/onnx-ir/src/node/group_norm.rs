//! # GroupNormalization
//!
//! Group normalization operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GroupNormalization.html>
//!
//! ## Attributes
//! - `epsilon` (float, default=1e-5): Numerical stability constant
//! - `num_groups` (int, required): Number of groups
//! - `stash_type` (int, optional): Intermediate calculation precision
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x H x W)
//! - `scale` (T): Scale tensor (C)
//! - `bias` (T): Bias tensor (C)
//!
//! ## Outputs
//! - `Y` (T): Normalized output tensor
//!
//! ## Opset Versions
//! - Opset 18+

use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for GroupNorm operations
#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Number of groups
    pub num_groups: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Whether to use full precision for intermediate calculations (stash_type == 1)
    pub full_precision: bool,
}

impl NodeConfig for GroupNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

impl GroupNormConfig {
    /// Create a new GroupNormConfig
    pub fn new(num_features: usize, num_groups: usize, epsilon: f64, full_precision: bool) -> Self {
        Self {
            num_features,
            num_groups,
            epsilon,
            full_precision,
        }
    }
}

pub struct GroupNormProcessor;

impl NodeProcessor for GroupNormProcessor {
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
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        const MIN: usize = 18;

        crate::processor::validate_opset(opset, MIN)?;
        crate::processor::validate_min_inputs(node, 3)?;
        crate::processor::validate_output_count(node, 1)?;

        // Validate attributes before extracting config
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" | "num_groups" | "stash_type" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for GroupNorm: {key}"),
                    });
                }
            }
        }

        // Validate num_groups divisibility
        let config = node.config::<GroupNormConfig>();

        if config.num_groups > 0 && !config.num_features.is_multiple_of(config.num_groups) {
            return Err(ProcessError::Custom(
                "GroupNorm: number of features must be divisible by the number of groups"
                    .to_string(),
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
            .as_ref()
            .ok_or_else(|| {
                ProcessError::Custom("GroupNorm: weight tensor must be present".to_string())
            })?
            .shape
            .clone();

        let num_features = weight_shape[0];
        let mut num_groups = None;
        let mut epsilon = 1e-5;
        let mut stash_type = 1; // Default value is 1 (full precision)

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" => epsilon = value.clone().into_f32(),
                "num_groups" => num_groups = Some(value.clone().into_i64() as usize),
                "stash_type" => stash_type = value.clone().into_i64(),
                _ => {}
            }
        }

        let num_groups = num_groups.ok_or_else(|| {
            ProcessError::MissingAttribute(
                "GroupNorm: num_groups attribute must be present".to_string(),
            )
        })?;

        let full_precision = stash_type == 1;
        let config = GroupNormConfig::new(num_features, num_groups, epsilon as f64, full_precision);
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
        num_features: usize,
        num_groups: usize,
        stash_type: i64,
    ) -> NodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::GroupNormalization, "test_groupnorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_int("num_groups", num_groups as i64)
            .attr_int("stash_type", stash_type)
            .attr_float("epsilon", epsilon)
    }

    #[test]
    fn test_group_norm_config_basic() {
        let mut node = create_test_node(1e-5, 64, 8, 1).build_with_graph_data(18);
        let processor = GroupNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 18).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 18, &prefs).unwrap();

        let config = node.config::<GroupNormConfig>();
        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(config.full_precision); // stash_type == 1
    }

    #[test]
    fn test_group_norm_config_no_stash_type() {
        let mut node = create_test_node(1e-5, 64, 8, 0).build_with_graph_data(18);
        let processor = GroupNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 18).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 18, &prefs).unwrap();

        let config = node.config::<GroupNormConfig>();
        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(!config.full_precision); // stash_type == 0
    }

    #[test]
    fn test_group_norm_config_invalid_num_groups() {
        // num features is not divisible by num groups
        let mut node = create_test_node(1e-5, 64, 7, 0).build_with_graph_data(18);
        let processor = GroupNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 18).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 18, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
