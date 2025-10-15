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
        // Validate opset
        if opset < 1 {
            return Err(ProcessError::UnsupportedOpset {
                required: 1,
                actual: opset,
            });
        }

        // Validate input count
        if node.inputs.len() != 2 {
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

        log::debug!("MatMul rank inference for node {}", node.name);

        match (&node.inputs[0].ty, &node.inputs[1].ty) {
            (ArgType::Tensor(a), ArgType::Tensor(b)) => {
                log::debug!(
                    "MatMul input ranks for {}: a.rank={}, b.rank={}",
                    node.name,
                    a.rank,
                    b.rank
                );

                let mut out_rank = max(a.rank, b.rank);
                if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                    out_rank -= 1;
                    log::debug!(
                        "MatMul special case for node {}: reducing output rank",
                        node.name
                    );
                }

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: a.elem_type.clone(),
                    rank: out_rank,
                    static_shape: None,
                });

                log::debug!("MatMul output rank for {}: {}", node.name, out_rank);
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
    use crate::ir::{ElementType, NodeType};
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_matmul_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let processor = MatMulProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
