//! # RandomLike Operations (RandomUniformLike, RandomNormalLike)
//!
//! Generates random tensors with the same shape as the input tensor, drawing values
//! from either a uniform or normal distribution.
//!
//! **ONNX Specs**:
//! - RandomUniformLike: <https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html>
//! - RandomNormalLike: <https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html>
//!
//! ## Type Constraints
//! - T1: Any tensor type
//! - T2: float16, float32, or float64 tensor
//!
//! ## Opset Versions
//! - Available since opset version 1
//! - Current version: 22
use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::Argument;

use crate::ir::{ArgType, DType, Node, NodeBuilder, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

/// Configuration for RandomNormalLike operation.
#[derive(Debug, Clone)]
pub struct RandomNormalLikeConfig {
    pub mean: f64,
    pub scale: f64,
}

/// Configuration for RandomUniformLike operation.
#[derive(Debug, Clone)]
pub struct RandomUniformLikeConfig {
    pub low: f64,
    pub high: f64,
}

/// Enum config that can hold either RandomNormalLike or RandomUniformLike config
#[derive(Debug, Clone)]
pub enum RandomLikeConfig {
    Normal(RandomNormalLikeConfig),
    Uniform(RandomUniformLikeConfig),
}

/// Node representation for RandomNormalLike operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct RandomNormalLikeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RandomNormalLikeConfig,
}

/// Node representation for RandomUniformLike operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct RandomUniformLikeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RandomUniformLikeConfig,
}

pub(crate) struct RandomLikeProcessor;

impl NodeProcessor for RandomLikeProcessor {
    type Config = RandomLikeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
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
        // TODO: Spec mentions RandomNormalLike has mean/scale attributes, RandomUniformLike has high/low
        // These attributes are documented but not validated or extracted into config
        // TODO: Spec also mentions 'seed' attribute that is not validated

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
            .unwrap_or(DataType::FLOAT);

        let elem_type = match dtype {
            DataType::FLOAT => DType::F32,
            DataType::FLOAT16 => DType::F16,
            DataType::DOUBLE => DType::F64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("Tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: elem_type,
                rank: tensor.rank,
                static_shape: tensor.static_shape.clone(),
            });

            Ok(())
        } else {
            Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: "Only tensor input is valid".to_string(),
            })
        }
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        let config = match node.node_type {
            crate::ir::NodeType::RandomNormalLike => {
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
                RandomLikeConfig::Normal(RandomNormalLikeConfig { mean, scale })
            }
            crate::ir::NodeType::RandomUniformLike => {
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
                RandomLikeConfig::Uniform(RandomUniformLikeConfig { low, high })
            }
            _ => {
                return Err(ProcessError::Custom(format!(
                    "RandomLikeProcessor does not support node type {:?}",
                    node.node_type
                )));
            }
        };

        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        match config {
            RandomLikeConfig::Normal(normal_like_config) => {
                Node::RandomNormalLike(RandomNormalLikeNode {
                    name: builder.name,
                    inputs: builder.inputs,
                    outputs: builder.outputs,
                    config: normal_like_config,
                })
            }
            RandomLikeConfig::Uniform(uniform_like_config) => {
                Node::RandomUniformLike(RandomUniformLikeNode {
                    name: builder.name,
                    inputs: builder.inputs,
                    outputs: builder.outputs,
                    config: uniform_like_config,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(
        dtype: i32,
        input_rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> NodeBuilder {
        TestNodeBuilder::new(NodeType::RandomNormalLike, "test_random_like")
            .input_tensor_f32("input", input_rank, static_shape)
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .build()
    }

    #[test]
    fn test_random_like_float() {
        let mut node = create_test_node(DataType::FLOAT.value(), 3, None);
        let processor = RandomLikeProcessor;
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
    fn test_random_like_double() {
        let mut node = create_test_node(DataType::DOUBLE.value(), 2, Some(vec![5, 10]));
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![5, 10]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_like_invalid_input() {
        let mut node = create_test_node(DataType::FLOAT.value(), 2, None);
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_random_like_unsupported_type() {
        let mut node = create_test_node(DataType::INT32.value(), 2, None);
        let processor = RandomLikeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
