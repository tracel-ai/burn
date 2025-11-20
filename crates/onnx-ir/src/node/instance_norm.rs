//! # InstanceNormalization
//!
//! Applies instance normalization to the input as described in
//! <https://arxiv.org/abs/1607.08022>.
//!
//! The operation normalizes each channel in each data instance independently:
//! `y = scale * (x - mean) / sqrt(variance + epsilon) + B`, where mean and
//! variance are computed per instance per channel.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html>
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1-5**: Earlier versions with different epsilon handling
//! - **Opset 6+**: Current version with epsilon=1e-5 default and standardized behavior
//!
//! ## Missing Test Coverage
//! - TODO: No test for custom epsilon values (e.g., epsilon=1e-3) - Only default epsilon tested
//! - TODO: No test for edge cases: zero-mean inputs, constant inputs, single channel
//! - TODO: No test validating behavior with different batch sizes or spatial dimensions
use derive_new::new;
use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for InstanceNorm operations
#[derive(Debug, Clone, new)]
pub struct InstanceNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

/// Node representation for InstanceNormalization operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct InstanceNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: InstanceNormConfig,
}

pub(crate) struct InstanceNormProcessor;

impl NodeProcessor for InstanceNormProcessor {
    type Config = InstanceNormConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 6,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input tensor dtype is floating-point type - Type constraint T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16) not enforced - burn/crates/onnx-ir/src/node/instance_norm.rs:88
        // TODO: Validate that scale and bias tensors are 1D and have size C matching the channel dimension of input - Shape mismatch could cause runtime errors - burn/crates/onnx-ir/src/node/instance_norm.rs:88
        // TODO: Validate that input tensor is at least 3D (N x C x D1 ...) - Spec requires minimum rank of 3 - burn/crates/onnx-ir/src/node/instance_norm.rs:88

        // Validate attributes before extracting config
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for InstanceNorm: {key}"),
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
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("InstanceNorm: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        let num_features = weight_shape[0];
        let mut epsilon = 1e-5;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "epsilon" {
                // TODO: Validate epsilon > 0 for numerical stability - Negative or zero epsilon could cause division by zero or numerical issues - burn/crates/onnx-ir/src/node/instance_norm.rs:128
                epsilon = value.clone().into_f32()
            }
        }

        let config = InstanceNormConfig::new(num_features, epsilon as f64);
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::InstanceNormalization(InstanceNormalizationNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(epsilon: f32, num_features: usize) -> TestNodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        TestNodeBuilder::new(NodeType::InstanceNormalization, "test_instancenorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
    }

    #[test]
    fn test_instance_norm_config_basic() {
        let node = create_test_node(1e-5, 64).build_with_graph_data(16);
        let mut node = node;
        let processor = InstanceNormProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }
}
