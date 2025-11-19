//! # NonZero
//!
//! Returns indices of non-zero elements.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__NonZero.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version. Returns 2D tensor with shape [rank(X), num_non_zero].
//! - **Opset 13**: Added support for bfloat16 input type.
//!
//! ## Type Constraints (from ONNX spec)
//! - T: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16),
//!   tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double),
//!   tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
//!
//! TODO: Add validation for supported input types - current implementation accepts any tensor type
//! without validation against ONNX type constraints. While this may work for numeric types,
//! unsupported types like string, complex64, complex128 should be explicitly rejected.
//! Location: infer_types method after line 38

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, NodeBuilder, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for NonZero operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct NonZeroNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct NonZeroProcessor;

impl NodeProcessor for NonZeroProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
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
        // Note: Implementation correctly validates inputs/outputs per spec (1 input, 1 output)

        match &node.inputs[0].ty {
            ArgType::Tensor(_tensor) => {
                // TODO: Missing test coverage for zero-size input tensors (e.g., shape [0, 5] or [3, 0])
                // The spec allows zero-size tensors, but there's no test validating behavior when
                // input tensor has zero elements. Should return [rank, 0] shaped output.
                // Add test: nonzero_zero_size_tensor

                // Output is always a 2D Int64 tensor
                // Shape: [input_tensor_rank, num_nonzero_elements]
                // First dimension equals input tensor rank
                // Second dimension is dynamic (depends on data)
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 2,
                    static_shape: None, // Dynamic shape - second dimension depends on number of nonzero elements
                });
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        }

        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::NonZero(NonZeroNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_nonzero_update_output() {
        let mut node = TestNodeBuilder::new(NodeType::NonZero, "test_nonzero")
            .input_tensor_f32("input", 3, Some(vec![2, 3, 4]))
            .output_tensor_i64("output", 2, None) // rank will be updated
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_update_output_1d() {
        let mut node = TestNodeBuilder::new(NodeType::NonZero, "test_nonzero_1d")
            .input_tensor_i32("input", 1, Some(vec![5]))
            .output_tensor_i64("output", 2, None)
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_update_output_4d() {
        let mut node = TestNodeBuilder::new(NodeType::NonZero, "test_nonzero_4d")
            .input_tensor_f64("input", 4, Some(vec![2, 3, 4, 5]))
            .output_tensor_i64("output", 2, None)
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
