//! # NonZero
//!
//! Returns indices of non-zero elements.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__NonZero.html>
//!
//! ## Attributes
//! None
//!
//! ## Inputs
//! - `X` (T): Input tensor
//!
//! ## Outputs
//! - `Y` (int64): Indices tensor, shape \[rank(X), num_non_zero\]
//!
//! ## Opset Versions
//! - Opset 9+

use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct NonZeroProcessor;

impl NodeProcessor for NonZeroProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 9)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        match &node.inputs[0].ty {
            ArgType::Tensor(_tensor) => {
                // Output is always a 2D Int64 tensor
                // Shape: [input_tensor_rank, num_nonzero_elements]
                // First dimension equals input tensor rank
                // Second dimension is dynamic (depends on data)
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_nonzero_update_output() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero")
            .input_tensor_f32("input", 3, Some(vec![2, 3, 4]))
            .output_tensor_i64("output", 2, None) // rank will be updated
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_update_output_1d() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero_1d")
            .input_tensor_i32("input", 1, Some(vec![5]))
            .output_tensor_i64("output", 2, None)
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_nonzero_update_output_4d() {
        let mut node = NodeBuilder::new(NodeType::NonZero, "test_nonzero_4d")
            .input_tensor_f64("input", 4, Some(vec![2, 3, 4, 5]))
            .output_tensor_i64("output", 2, None)
            .build();

        let processor = NonZeroProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
