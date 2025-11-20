//! # MatMulInteger
//!
//! Matrix multiplication for quantized integer tensors with zero-point support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MatMulInteger.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version introducing quantized integer matrix multiplication with zero-point
//!   support. Outputs int32 results from int8/uint8 inputs.
//!
//! **Implementation Note**: This implementation validates opset 10+ (see line 37).
//! The spec allows 2-4 inputs (optional zero-point tensors), but implementation only validates minimum
//! of 2 inputs (see FIXME at line 44).

use crate::ir::{ArgType, Argument, DType, Node, NodeBuilder, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use core::cmp::max;

/// Node representation for MatMulInteger operation
#[derive(Debug, Clone)]
pub struct MatMulIntegerNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct MatMulIntegerProcessor;

impl NodeProcessor for MatMulIntegerProcessor {
    type Config = ();

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // MatMulInteger implementation supports opset 10+
        if opset < 10 {
            return Err(ProcessError::UnsupportedOpset {
                required: 10,
                actual: opset,
            });
        }

        // FIXME: Spec mentions 2-4 inputs (A, B, a_zero_point optional, b_zero_point optional)
        // but we only validate minimum 2 inputs. Should validate that we don't have more than 4 inputs
        // and that the optional zero_point inputs have correct types (T1/T2).

        // Validate input count
        if node.inputs.len() < 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: node.inputs.len(),
            });
        }

        // Validate output count
        if node.outputs.len() != 1 {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        match (&node.inputs[0].ty, &node.inputs[1].ty) {
            (ArgType::Tensor(a), ArgType::Tensor(b)) => {
                let mut out_rank = max(a.rank, b.rank);

                // Special cases: vector–matrix or matrix–vector reduces rank by 1
                if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                    out_rank -= 1;
                }

                // ONNX spec: MatMulInteger output is always int32
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: DType::I32,
                    rank: out_rank,
                    static_shape: None,
                });

                Ok(())
            }
            _ => Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: "MatMulInteger expects tensor inputs".to_string(),
            }),
        }
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::MatMulInteger(MatMulIntegerNode {
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

    fn create_test_node(a_rank: usize, b_rank: usize) -> NodeBuilder {
        TestNodeBuilder::new(NodeType::MatMulInteger, "test_matmulinteger")
            .input_tensor_i32("A", a_rank, None)
            .input_tensor_i32("B", b_rank, None)
            .output_tensor_i32("Y", 0, None) // rank will be updated
            .build()
    }

    #[test]
    fn test_update_outputs_standard_case() {
        let mut node = create_test_node(2, 2);
        let processor = MatMulIntegerProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_update_outputs_vector_matrix() {
        let mut node = create_test_node(1, 2);
        let processor = MatMulIntegerProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Scalar(DType::I32);
        let processor = MatMulIntegerProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
