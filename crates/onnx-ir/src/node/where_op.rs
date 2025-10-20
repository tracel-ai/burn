//! # Where
//!
//! Selects elements from X or Y based on condition (ternary operator).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Where.html>
//!
//! ## Attributes
//! None
//!
//! ## Inputs
//! - `condition` (B): Boolean condition tensor
//! - `X` (T): Values when condition is true
//! - `Y` (T): Values when condition is false
//!
//! ## Outputs
//! - `output` (T): Selected elements
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version with broadcasting support for all three inputs.

use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::processor::{
    NodeProcessor, OutputPreferences, ProcessError, compute_broadcast_rank,
    compute_broadcast_static_shape,
};

/// Get element type from ArgType, handling Shape types specially
fn get_elem_type(arg_type: &ArgType) -> ElementType {
    match arg_type {
        ArgType::Scalar(elem_type) => elem_type.clone(),
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        ArgType::Shape(_) => ElementType::Int64, // Shape types are always i64
    }
}

/// Check if output should be a Shape type
fn should_output_shape(
    x: &ArgType,
    y: &ArgType,
    output_rank: usize,
    elem_type: &ElementType,
) -> bool {
    // Output Shape if both inputs are Shape and output would be 1D int64
    matches!(x, ArgType::Shape(_))
        && matches!(y, ArgType::Shape(_))
        && output_rank == 1
        && *elem_type == ElementType::Int64
}

/// Get size of Shape type, or 1 for other types
fn get_shape_size(arg_type: &ArgType) -> usize {
    match arg_type {
        ArgType::Shape(size) => *size,
        _ => 1,
    }
}

/// Update output type for Where operation.
///
pub struct WhereProcessor;

impl NodeProcessor for WhereProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 9)?;
        crate::processor::validate_input_count(node, 3)?;
        crate::processor::validate_output_count(node, 1)?;

        let condition = &node.inputs[0].ty;
        let x = &node.inputs[1].ty;
        let y = &node.inputs[2].ty;

        // Get element types, handling Shape types specially
        let x_elem_type = get_elem_type(x);
        let y_elem_type = get_elem_type(y);
        let condition_elem_type = get_elem_type(condition);

        if !matches!(condition, ArgType::Shape(_)) && condition_elem_type != ElementType::Bool {
            return Err(ProcessError::TypeMismatch {
                expected: "Bool".to_string(),
                actual: format!("{:?}", condition_elem_type),
            });
        }

        let elem_type = if x_elem_type == y_elem_type {
            x_elem_type
        } else if matches!(x, ArgType::Shape(_)) {
            y_elem_type
        } else if matches!(y, ArgType::Shape(_)) {
            x_elem_type
        } else {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{:?}", x_elem_type),
                actual: format!("{:?}", y_elem_type),
            });
        };

        let output_rank = compute_broadcast_rank(&node.inputs);

        // Determine output type
        if output_rank == 0 {
            node.outputs[0].ty = ArgType::Scalar(elem_type);
        } else if should_output_shape(x, y, output_rank, &elem_type) {
            // If both inputs are Shape types and output is 1D int64, preserve Shape type
            let shape_size = get_shape_size(x).max(get_shape_size(y));
            node.outputs[0].ty = ArgType::Shape(shape_size);
        } else {
            // Try to propagate static shape using the shared broadcast helper
            let static_shape = compute_broadcast_static_shape(&node.inputs);

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type,
                rank: output_rank,
                static_shape,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(condition_rank: usize, x_rank: usize, y_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", condition_rank, None)
            .input_tensor_f32("X", x_rank, None)
            .input_tensor_f32("Y", y_rank, None)
            .output_tensor_f32("output", 0, None) // Rank will be updated
            .build()
    }

    #[test]
    fn test_where_basic() {
        let mut node = create_test_node(2, 3, 2);
        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3); // max(2, max(3, 2)) = 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_where_scalar_result() {
        let mut node = create_test_node(0, 0, 0);
        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_where_invalid_condition() {
        let mut node = create_test_node(2, 2, 2);

        // Replace condition with non-boolean tensor
        let non_bool_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("x", 2, None)
            .build()
            .inputs
            .pop()
            .unwrap();

        node.inputs[0] = non_bool_input;
        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_where_mismatched_types() {
        let mut node = create_test_node(2, 2, 2);

        // Replace Y with int64 tensor (different from X's float32)
        let int64_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_i64("y", 2, None)
            .build()
            .inputs
            .pop()
            .unwrap();

        node.inputs[2] = int64_input;
        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_where_with_shape_inputs() {
        let mut node = create_test_node(1, 0, 0);

        // Replace X and Y with Shape types
        node.inputs[1].ty = ArgType::Shape(3);
        node.inputs[2].ty = ArgType::Shape(3);

        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(size) => {
                assert_eq!(*size, 3); // Should preserve Shape type
            }
            _ => panic!("Expected Shape output"),
        }
    }

    #[test]
    fn test_where_static_shape_propagation() {
        // Test that static shapes propagate correctly through Where
        let mut node = NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", 2, Some(vec![2, 2]))
            .input_tensor_f32("X", 2, Some(vec![2, 2]))
            .input_tensor_f32("Y", 2, Some(vec![2, 2]))
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![2, 2]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_where_static_shape_propagation_partial() {
        // Test that static shape propagates even when only some inputs have it
        let mut node = NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", 2, None) // No static shape
            .input_tensor_f32("X", 2, Some(vec![3, 4])) // Has static shape
            .input_tensor_f32("Y", 2, Some(vec![3, 4])) // Has static shape
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = WhereProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                // Since X and Y have the same static shape, it should propagate
                assert_eq!(tensor.static_shape, Some(vec![3, 4]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
