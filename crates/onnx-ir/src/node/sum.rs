//! # Sum
//!
//! Element-wise sum of each of the input tensors with multidirectional (Numpy-style)
//! broadcasting support. All inputs and outputs must have the same data type.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Sum.html>
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

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Sum operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct SumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Sum operation
pub(crate) struct SumProcessor;

impl NodeProcessor for SumProcessor {
    type Config = ();

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
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing validation of input count upper bound - ONNX spec allows up to 2^31-1 inputs.
        // While this is huge, there should be some practical limit or at least documentation.

        let mut expected_dtype: Option<DType> = None;
        for input in &node.inputs {
            let current_dtype = match &input.ty {
                ArgType::Tensor(tensor) => tensor.dtype,
                ArgType::Scalar(dtype) => *dtype,
                ArgType::Shape(_) => DType::I64,
            };

            if !current_dtype.is_float() && !current_dtype.is_int() && !current_dtype.is_uint() {
                return Err(ProcessError::TypeMismatch {
                    expected: "Numeric (Float, Int, or UInt)".to_string(),
                    actual: format!("{:?}", current_dtype),
                });
            }

            if let Some(expected) = expected_dtype {
                if current_dtype != expected {
                    return Err(ProcessError::TypeMismatch {
                        expected: format!("{:?}", expected),
                        actual: format!("{:?}", current_dtype),
                    });
                }
            } else {
                expected_dtype = Some(current_dtype);
            }
        }

        // TODO: Missing validation that all inputs have broadcastable shapes.
        // While same_as_input_broadcast handles inference, it doesn't validate compatibility.

        same_as_input_broadcast(node);

        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Sum(SumNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, NodeType, TensorType};

    #[test]
    fn test_sum_processor_two_inputs() {
        let processor = SumProcessor;

        let mut node = crate::ir::RawNode {
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

        let mut node = crate::ir::RawNode {
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
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_sum_processor_single_input() {
        let processor = SumProcessor;

        let mut node = crate::ir::RawNode {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 4,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 4),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_sum_processor_broadcasting() {
        let processor = SumProcessor;

        let mut node = crate::ir::RawNode {
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
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "c".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 1,
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
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_sum_processor_type_constraint_validation() {
        let processor = SumProcessor;

        let mut node = crate::ir::RawNode {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::Bool,
                        rank: 1,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::Bool,
                        rank: 1,
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
        };

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);

        assert!(
            result.is_err(),
            "infer_types should fail/reject non-numeric inputs such as bool"
        );
        match result.unwrap_err() {
            ProcessError::TypeMismatch { expected, actual } => {
                assert_eq!(expected, "Numeric (Float, Int, or UInt)");
                assert_eq!(actual, "Bool");
            }
            err => panic!("Expected TypeMismatch error for non-numeric type, got {:?}", err),
        }
    }

    #[test]
    fn test_sum_processor_zero_size_input() {
        let processor = SumProcessor;

        let mut node = crate::ir::RawNode {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 3,
                        static_shape: Some(vec![0, 3, 4]),
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 3,
                        static_shape: Some(vec![0, 3, 4]),
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
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Verify the output preserves the shape [0, 3, 4]
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3, "Rank should be preserved");
                assert_eq!(
                    t.static_shape,
                    Some(vec![0, 3, 4]),
                    "Output shape should be [0, 3, 4]"
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_sum_processor_many_inputs() {
        let processor = SumProcessor;
        let num_inputs = 250;
        let common_shape = vec![10, 5];

        let inputs: Vec<Argument> = (0..num_inputs)
            .map(|i| Argument {
                name: format!("input_{}", i),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: Some(common_shape.clone()),
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            })
            .collect();

        let mut node = crate::ir::RawNode {
            node_type: NodeType::Sum,
            name: "test_sum_many".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
        };

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2, "Rank should be preserved");
                assert_eq!(
                    t.static_shape,
                    Some(common_shape),
                    "Output shape should match input shape"
                );
                assert_eq!(t.dtype, DType::F32, "Data type should be preserved");
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
