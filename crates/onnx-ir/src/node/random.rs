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

use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

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
            DataType::FLOAT => ElementType::Float32,
            DataType::DOUBLE => ElementType::Float64,
            _ => {
                return Err(ProcessError::InvalidAttribute {
                    name: "dtype".to_string(),
                    reason: format!("tensor with type {dtype:?} not supported for random output"),
                });
            }
        };

        let rank = shape.len();

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn crate::ir::NodeConfig>>, ProcessError> {
        // Random has no config
        Ok(None)
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
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
                assert_eq!(tensor.elem_type, ElementType::Float64);
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
