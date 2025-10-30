//! # Random Tensor Generation (RandomNormal, RandomUniform)
//!
//! This module implements processors for ONNX random tensor generation operations that create
//! tensors with random values sampled from specified distributions.
//!
//! **ONNX Specs**:
//! - RandomNormal: <https://onnx.ai/onnx/operators/onnx__RandomNormal.html>
//! - RandomUniform: <https://onnx.ai/onnx/operators/onnx__RandomUniform.html>
//!
//! ## Supported Operations
//!
//! - **RandomNormal**: Generates a tensor with values drawn from a normal distribution
//! - **RandomUniform**: Generates a tensor with values drawn from a uniform distribution
//!
//! Both operations require a `shape` attribute and support an optional `dtype` attribute
//! (defaults to FLOAT if not specified).
//!
//! ## Opset Versions
//!
//! ### RandomNormal
//! - **Opset 1**: Initial version with shape, dtype, mean, scale, and seed attributes.
//!
//! ### RandomUniform
//! - **Opset 1**: Initial version with shape, dtype, high, low, and seed attributes.

use crate::ir::{ArgType, DType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;
use std::any::Any;

/// Configuration for RandomNormal operation.
#[derive(Debug, Clone)]
pub struct RandomNormalConfig {
    pub mean: f64,
    pub scale: f64,
    pub shape: Vec<usize>,
}

impl NodeConfig for RandomNormalConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Configuration for RandomUniform operation.
#[derive(Debug, Clone)]
pub struct RandomUniformConfig {
    pub low: f64,
    pub high: f64,
    pub shape: Vec<usize>,
}

impl NodeConfig for RandomUniformConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct RandomProcessor;

impl NodeProcessor for RandomProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Validate that this node has zero inputs (Random operations don't take inputs)
        // Random operators should have no inputs, but this isn't validated.
        // Should add: validate_input_count(node, 0) or validate_max_inputs(node, 0)
        // Location: After validate_output_count

        // TODO: RandomNormal mean and scale attributes not validated or extracted
        // ONNX spec defines:
        // - mean (float, default=0.0): Mean of the normal distribution
        // - scale (float, default=1.0): Standard deviation of the normal distribution
        // - seed (float, optional): Random seed for reproducibility
        // These attributes exist in spec but are completely ignored by implementation.
        // Should extract into config and validate ranges (scale > 0).
        // Location: extract_config method

        // TODO: RandomUniform high/low attributes not validated or extracted
        // ONNX spec defines:
        // - high (float, default=1.0): Upper boundary of uniform distribution (exclusive)
        // - low (float, default=0.0): Lower boundary of uniform distribution (inclusive)
        // - seed (float, optional): Random seed for reproducibility
        // These attributes are ignored. Should extract into config and validate low < high.
        // Location: extract_config method

        // TODO: Missing test coverage for non-default distribution parameters
        // Tests only validate basic shape output, not distribution parameters (mean, scale, high, low).
        // Add tests: random_normal_custom_mean_scale, random_uniform_custom_range
        // Note: Testing actual distribution properties is hard, but should at least extract attrs.

        // TODO: Missing validation for seed attribute
        // Spec mentions seed attribute for reproducibility but it's not extracted or validated.
        // Add tests: random_normal_with_seed, random_uniform_with_seed

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
            .unwrap_or(DataType::FLOAT);

        let shape = node
            .attrs
            .get("shape")
            .ok_or_else(|| ProcessError::Custom("required shape attribute missing".to_string()))?
            .clone()
            .into_i64s();

        let elem_type = match dtype {
            DataType::FLOAT => DType::F32,
            DataType::DOUBLE => DType::F64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        let rank = shape.len();

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: elem_type,
            rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn crate::ir::NodeConfig>>, ProcessError> {
        let shape = node
            .attrs
            .get("shape")
            .ok_or_else(|| ProcessError::Custom("required shape attribute missing".to_string()))?
            .clone()
            .into_i64s();
        let shape: Vec<usize> = shape.into_iter().map(|i| i as usize).collect();

        let config: Box<dyn NodeConfig> = match node.node_type {
            crate::ir::NodeType::RandomNormal => {
                let mean = node
                    .attrs
                    .get("mean")
                    .map(|v| v.clone().into_f32() as f64)
                    .unwrap_or(0.0);
                let scale = node
                    .attrs
                    .get("scale")
                    .map(|v| v.clone().into_f32() as f64)
                    .unwrap_or(1.0);
                Box::new(RandomNormalConfig { mean, scale, shape })
            }
            crate::ir::NodeType::RandomUniform => {
                let low = node
                    .attrs
                    .get("low")
                    .map(|v| v.clone().into_f32() as f64)
                    .unwrap_or(0.0);
                let high = node
                    .attrs
                    .get("high")
                    .map(|v| v.clone().into_f32() as f64)
                    .unwrap_or(1.0);
                Box::new(RandomUniformConfig { low, high, shape })
            }
            _ => {
                return Err(ProcessError::Custom(format!(
                    "RandomProcessor does not support node type {:?}",
                    node.node_type
                )));
            }
        };

        Ok(Some(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, shape: Vec<i64>) -> Node {
        NodeBuilder::new(NodeType::RandomNormal, "test_random")
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .attr_ints("shape", shape)
            .build()
    }

    #[test]
    fn test_random_normal_float() {
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3, 4]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_double() {
        let mut node = create_test_node(DataType::DOUBLE.value(), vec![5]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_missing_shape() {
        // Create node and then manually remove the shape attribute
        let mut node = create_test_node(DataType::FLOAT.value(), vec![2, 3]);
        node.attrs.remove("shape");
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_random_normal_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), vec![2, 3]);
        let processor = RandomProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
