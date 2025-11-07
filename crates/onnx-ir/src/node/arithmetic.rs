//! # Arithmetic Operations (Add, Sub, Mul, Div)
//!
//! Implements element-wise binary arithmetic operations with multidirectional (Numpy-style)
//! broadcasting support. These operators share the same type propagation semantics and handle
//! special cases for Shape and Scalar types to preserve semantics through arithmetic operations.
//!
//! **ONNX Specs**:
//! - Add: <https://onnx.ai/onnx/operators/onnx__Add.html>
//! - Sub: <https://onnx.ai/onnx/operators/onnx__Sub.html>
//! - Mul: <https://onnx.ai/onnx/operators/onnx__Mul.html>
//! - Div: <https://onnx.ai/onnx/operators/onnx__Div.html>
//!
//! ## Attributes
//!
//! None - these operators have no attributes.
//!
//! ## Inputs
//!
//! - `A` (T): First operand, any rank
//! - `B` (T): Second operand, any rank (must be type-compatible with A)
//!
//! ## Outputs
//!
//! - `C` (T): Result of element-wise operation (supports multidirectional broadcasting)
//!
//! ## Type Constraints
//!
//! T: Numeric tensor types (float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64)
//!
//! ## Opset Versions
//! - **Opset 1-6**: Limited broadcast support (unidirectional only)
//! - **Opset 7-12**: Added multidirectional (Numpy-style) broadcasting for Add, Sub, Div
//! - **Opset 13**: Multidirectional broadcasting for Mul, extended type support (bfloat16)
//! - **Opset 14+**: Extended type support to include uint8, int8, uint16, int16, uint32, uint64
//!
//! ## Special Handling
//!
//! This implementation includes type preference propagation for:
//! - **Shape arithmetic**: When operating on Shape types with constants, prefers constants as Shape
//! - **Scalar arithmetic**: When operating on Scalar types with constants, prefers constants as Scalar

use crate::ir::Node;
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError, same_as_input_broadcast,
};

/// Node processor for basic arithmetic binary operations
///
/// Handles type inference for element-wise arithmetic operations with special support for:
/// - Shape arithmetic (e.g., adding offsets to tensor shapes, dividing shapes)
/// - Scalar arithmetic (preserving scalar types through operations)
/// - Standard tensor broadcasting
///
/// This processor is used for Add, Sub, Mul, and Div operations as they all
/// share the same type propagation semantics.
pub struct ArithmeticBinaryProcessor;

impl NodeProcessor for ArithmeticBinaryProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 7,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() != 2 {
            return Ok(None);
        }

        let mut prefs = InputPreferences::new();

        // Type propagation for Shape arithmetic:
        // When performing arithmetic on a Shape with a constant, prefer the constant to be Shape type.
        // This is common in dynamic shape calculations like:
        // - new_shape = old_shape + offset
        // - half_shape = old_shape / 2

        // Case 1: Shape op Constant => prefer Constant as Shape
        if node.inputs[0].ty.is_shape() {
            prefs = prefs.add(&node.inputs[1].name, ArgPreference::Shape);
        }

        // Case 2: Constant op Shape => prefer Constant as Shape
        if node.inputs[1].ty.is_shape() {
            prefs = prefs.add(&node.inputs[0].name, ArgPreference::Shape);
        }

        // Type propagation for Scalar arithmetic:
        // When performing arithmetic on a Scalar with a constant, prefer the constant to be Scalar type.
        // This preserves scalar semantics through arithmetic operations.

        // Case 3: Scalar op Constant => prefer Constant as Scalar
        if node.inputs[0].ty.is_scalar() {
            prefs = prefs.add(&node.inputs[1].name, ArgPreference::Scalar);
        }

        // Case 4: Constant op Scalar => prefer Constant as Scalar
        if node.inputs[1].ty.is_scalar() {
            prefs = prefs.add(&node.inputs[0].name, ArgPreference::Scalar);
        }

        Ok(Some(prefs))
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Apply standard broadcasting rules to infer output type
        same_as_input_broadcast(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_arithmetic_add() {
        let node = NodeBuilder::new(NodeType::Add, "test_add")
            .input_tensor_f32("a", 2, None)
            .input_tensor_f32("b", 2, None)
            .output_default("c")
            .process(ArithmeticBinaryProcessor, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_sub() {
        let node = NodeBuilder::new(NodeType::Sub, "test_sub")
            .input_tensor_f32("a", 3, None)
            .input_tensor_f32("b", 3, None)
            .output_default("c")
            .process(ArithmeticBinaryProcessor, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_mul() {
        let node = NodeBuilder::new(NodeType::Mul, "test_mul")
            .input_tensor_f32("a", 4, None)
            .input_tensor_f32("b", 4, None)
            .output_default("c")
            .process(ArithmeticBinaryProcessor, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 4),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_div() {
        let node = NodeBuilder::new(NodeType::Div, "test_div")
            .input_tensor_f32("a", 2, None)
            .input_tensor_f32("b", 2, None)
            .output_default("c")
            .process(ArithmeticBinaryProcessor, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_invalid_opset() {
        let node = NodeBuilder::new(NodeType::Add, "test_add")
            .input_tensor_f32("a", 2, None)
            .input_tensor_f32("b", 2, None)
            .output_default("c")
            .build();

        let processor = ArithmeticBinaryProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 6, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::UnsupportedOpset {
                required: 7,
                actual: 6
            })
        ));
    }

    #[test]
    fn test_invalid_input_count() {
        let node = NodeBuilder::new(NodeType::Add, "test_add")
            .input_tensor_f32("a", 2, None)
            .output_default("c")
            .build();

        let processor = ArithmeticBinaryProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }
}
