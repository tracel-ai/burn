//! # MatMul
//!
//! Matrix product that behaves like numpy.matmul.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MatMul.html>
//!
//! **Note**: In the node_conversion phase, MatMul nodes with constant weights may be converted to Linear nodes.
//!
//! ## Attributes (None)
//!
//! ## Inputs
//! - `A` (T): N-dimensional matrix A
//! - `B` (T): N-dimensional matrix B
//!
//! ## Outputs
//! - `Y` (T): Matrix multiply results from A * B
//!
//! ## Type Constraints
//! - T: tensor(float), tensor(double), tensor(float16), tensor(bfloat16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with numpy.matmul-like behavior and broadcasting support.
//! - **Opset 9**: Added support for additional integer types (int32, int64, uint32, uint64).
//! - **Opset 13**: Added bfloat16 type support; no functional changes to operation semantics.
//!
//! **Implementation Note**: This implementation validates opset 1+.

use crate::ir::{ArgType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use core::cmp::max;

pub struct MatMulProcessor;

impl NodeProcessor for MatMulProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_input_count(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // Validate that no unexpected attributes are present
        // The spec states "Attributes (None)" for MatMul
        if let Some((key, _value)) = node.attrs.iter().next() {
            return Err(ProcessError::InvalidAttribute {
                name: key.clone(),
                reason: format!("MatMul does not accept any attributes, found: {}", key),
            });
        }

        match (&node.inputs[0].ty, &node.inputs[1].ty) {
            (ArgType::Tensor(a), ArgType::Tensor(b)) => {
                // Validate dtype compatibility - both inputs must have the same dtype
                if a.dtype != b.dtype {
                    return Err(ProcessError::TypeMismatch {
                        expected: format!("Both inputs to have dtype {:?}", a.dtype),
                        actual: format!(
                            "Input A has dtype {:?}, Input B has dtype {:?}",
                            a.dtype, b.dtype
                        ),
                    });
                }

                // Validate rank constraints per ONNX spec
                // MatMul requires both inputs to be at least 1D
                if a.rank < 1 || b.rank < 1 {
                    return Err(ProcessError::Custom(format!(
                        "MatMul requires both inputs to have rank >= 1, got ranks: A={}, B={}",
                        a.rank, b.rank
                    )));
                }

                // For matrix multiplication, the last dimension of A must match the first/last dimension of B
                // This is a basic shape compatibility check based on ranks
                // When both are 2D: A[M, K] x B[K, N] -> valid
                // When A is 1D and B is 2D: A[K] x B[K, N] -> valid
                // When A is 2D and B is 1D: A[M, K] x B[K] -> valid
                // Higher dimensional cases use broadcasting on batch dimensions

                let mut out_rank = max(a.rank, b.rank);
                if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                    out_rank -= 1;
                }

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: a.dtype,
                    rank: out_rank,
                    static_shape: None,
                });
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}, {:?}", node.inputs[0].ty, node.inputs[1].ty),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(a_rank: usize, b_rank: usize) -> Node {
        NodeBuilder::new(NodeType::MatMul, "test_matmul")
            .input_tensor_f32("A", a_rank, None)
            .input_tensor_f32("B", b_rank, None)
            .output_tensor_f32("C", 0, None) // Rank will be updated
            .build()
    }

    #[test]
    fn test_matmul_standard_case() {
        let mut node = create_test_node(2, 2);
        let processor = MatMulProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_matmul_broadcasting() {
        let mut node = create_test_node(3, 2);
        let processor = MatMulProcessor;

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
    fn test_matmul_vector_matrix() {
        // When multiplying a vector (rank 1) by a matrix (rank 2)
        // the result should have rank 1 (vector)
        let mut node = create_test_node(1, 2);
        let processor = MatMulProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_matmul_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
        let processor = MatMulProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
