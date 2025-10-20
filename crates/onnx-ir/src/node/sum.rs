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
    NodeProcessor, OutputPreferences, ProcessError, same_as_input_broadcast, validate_min_inputs,
    validate_opset, validate_output_count,
};

/// Node processor for Sum operation
pub struct SumProcessor;

impl NodeProcessor for SumProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        validate_opset(opset, 8)?;

        // Validate we have at least one input
        validate_min_inputs(node, 1)?;

        // Validate output count
        validate_output_count(node, 1)?;

        same_as_input_broadcast(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

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
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                data_id: None,
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
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "c".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                data_id: None,
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
}
