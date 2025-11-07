//! # Sum
//!
//! Element-wise sum of each of the input tensors with multidirectional (Numpy-style)
//! broadcasting support. All inputs and outputs must have the same data type.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Sum.html>
//!
//! ## Attributes
//!
//! None - this operator has no attributes.
//!
//! ## Inputs
//!
//! - Variable number of input tensors: `data_0`, `data_1`, ..., `data_N` (T)
//!   - Minimum: 1 input
//!   - Maximum: 2147483647 inputs
//!   - All inputs must have compatible shapes for broadcasting
//!
//! ## Outputs
//!
//! - `sum` (T): Element-wise sum of all input tensors (supports multidirectional broadcasting)
//!
//! ## Type Constraints
//!
//! T: Numeric tensor types (bfloat16, float16, float32, float64)
//!
//! ## Opset Versions
//!
//! - **Opset 1-5**: Basic element-wise sum
//! - **Opset 6-7**: Improved broadcasting support
//! - **Opset 8**: Multidirectional (Numpy-style) broadcasting
//! - **Opset 13**: Extended type support including bfloat16

use crate::ir::Node;
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node processor for Sum operation
pub struct SumProcessor;

impl NodeProcessor for SumProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 8,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing validation of input count upper bound - ONNX spec allows up to 2^31-1 inputs.
        // While this is huge, there should be some practical limit or at least documentation.

        // TODO: Missing validation that all inputs have compatible dtypes.
        // ONNX spec requires all inputs to have the same data type, but this isn't validated.

        // TODO: Missing validation that all inputs have broadcastable shapes.
        // While same_as_input_broadcast handles inference, it doesn't validate compatibility.

        same_as_input_broadcast(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, NodeType, TensorType};

    #[test]
    fn test_sum_processor_two_inputs() {
        let processor = SumProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_sum_processor_multiple_inputs() {
        let processor = SumProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "c".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    // TODO: Missing test for single input - Sum with one input should return that input unchanged.
    // This is a valid edge case per ONNX spec.

    // TODO: Missing test for broadcasting with different ranks.
    // E.g., Sum of [3, 4, 5] + [1, 5] + [5] should broadcast correctly.

    // TODO: Missing test for type constraint validation.
    // Sum should only support numeric types, need test to verify string/bool inputs are rejected.

    // TODO: Missing test for zero-size tensor inputs.
    // What happens with Sum of tensors with shape [0, 3, 4]?

    // TODO: Missing test for very many inputs (e.g., 100+ inputs).
    // Verify implementation can handle many inputs without stack overflow or performance issues.
}
