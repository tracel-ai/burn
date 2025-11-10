//! # Bernoulli
//!
//! Draws binary random numbers (0 or 1) from a Bernoulli distribution.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Bernoulli.html>
//!
//! ## Opset Versions
//!
//! - **Opset 15**: Initial version with dtype and seed attributes for drawing binary random numbers

use crate::ir::{ArgType, DType, Node, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::protos::tensor_proto::DataType;
use protobuf::Enum;

pub struct BernoulliProcessor;

impl NodeProcessor for BernoulliProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 15,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for unexpected attributes
        // TODO: Spec mentions 'seed' attribute but it's not validated or used in implementation

        // Get the tensor type and its rank
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };
        let rank = tensor.rank;
        let static_shape = tensor.static_shape.clone();

        // Infer elem type based on the dtype attribute
        let dtype = node
            .attrs
            .get("dtype")
            .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap());

        let elem_type = dtype.map_or(tensor.dtype, |dtype| match dtype {
            DataType::FLOAT => DType::F32,
            DataType::INT32 => DType::I32,
            DataType::INT64 => DType::I64,
            DataType::DOUBLE => DType::F64,
            DataType::BOOL => DType::Bool,
            _ => tensor.dtype, // Fallback to input type for unsupported dtype
        });

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: elem_type,
            rank,
            static_shape,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: Option<i32>, static_shape: Option<Vec<usize>>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Bernoulli, "test_bernoulli")
            .input_tensor_f32("input", 4, static_shape) // Rank 0 will be updated
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(dtype) = dtype {
            builder = builder.attr_int("dtype", dtype as i64)
        }

        builder.build()
    }

    #[test]
    fn test_bernoulli_int() {
        let mut node = create_test_node(Some(DataType::INT32.value()), Some(vec![3, 4, 2]));
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.static_shape, Some(vec![3, 4, 2]));
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_cast() {
        let mut node = create_test_node(None, Some(vec![3, 4, 2]));
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_no_static_shape() {
        let mut node = create_test_node(None, None);
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_bernoulli_invalid_input() {
        let mut node = create_test_node(Some(DataType::FLOAT.value()), None);
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
        let processor = BernoulliProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
