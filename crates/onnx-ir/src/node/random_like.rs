//! # RandomLike Operations (RandomUniformLike, RandomNormalLike)
//!
//! Generates random tensors with the same shape as the input tensor, drawing values
//! from either a uniform or normal distribution.
//!
//! **ONNX Specs**:
//! - RandomUniformLike: <https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html>
//! - RandomNormalLike: <https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html>
//!
//! ## Attributes
//!
//! ### Common Attributes
//! - `dtype` (int, optional): The data type for the elements of the output tensor.
//!   If not specified, uses the data type of the input tensor.
//!   - Supported types: float16, float32 (float), float64 (double)
//! - `seed` (float, optional): Seed to the random generator. If not specified, one
//!   will be auto-generated.
//!
//! ### RandomUniformLike Specific
//! - `high` (float, optional): Upper boundary of the uniform distribution. Default: 1.0
//! - `low` (float, optional): Lower boundary of the uniform distribution. Default: 0.0
//!
//! ### RandomNormalLike Specific
//! - `mean` (float, optional): The mean of the normal distribution. Default: 0.0
//! - `scale` (float, optional): The standard deviation of the normal distribution. Default: 1.0
//!
//! ## Inputs
//! - `input` (T1): Input tensor used to copy shape and optionally type information.
//!
//! ## Outputs
//! - `output` (T2): Random tensor with same shape as input, containing values drawn
//!   from the specified distribution (uniform or normal).
//!
//! ## Type Constraints
//! - T1: Any tensor type
//! - T2: float16, float32, or float64 tensor
//!
//! ## Opset Versions
//! - Available since opset version 1
//! - Current version: 22

use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

pub struct RandomLikeProcessor;

impl NodeProcessor for RandomLikeProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_min_inputs(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
            .unwrap_or(DataType::FLOAT);

        let elem_type = match dtype {
            DataType::FLOAT => ElementType::Float32,
            DataType::FLOAT16 => ElementType::Float16,
            DataType::DOUBLE => ElementType::Float64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("Tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type,
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
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn crate::ir::NodeConfig>>, ProcessError> {
        // RandomLike has no config
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, input_rank: usize, static_shape: Option<Vec<usize>>) -> Node {
        NodeBuilder::new(NodeType::RandomNormalLike, "test_random_like")
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
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
                assert_eq!(tensor.elem_type, ElementType::Float64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![5, 10]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_like_invalid_input() {
        let mut node = create_test_node(DataType::FLOAT.value(), 2, None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
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
