//! Processor for basic arithmetic operations (Add, Sub, Mul, Div)
//!
//! These operations perform element-wise arithmetic with broadcasting support.
//! Special handling for Shape and Scalar types to preserve semantics through arithmetic.

use crate::ir::Node;
use crate::processor::{InputPreferences, NodeProcessor, OutputPreferences, ProcessError};
use crate::util::same_as_input_broadcast;

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
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset version
        crate::util::validate_opset(opset, 7)?;

        // Validate input count
        crate::util::validate_input_count(node, 2)?;

        // Validate output count
        crate::util::validate_output_count(node, 1)?;

        // Apply standard broadcasting rules to infer output type
        same_as_input_broadcast(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_arithmetic_add() {
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Add,
            name: "test_add".to_string(),
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
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_sub() {
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Sub,
            name: "test_sub".to_string(),
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
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_mul() {
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Mul,
            name: "test_mul".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
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
                        rank: 4,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 4),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_div() {
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Div,
            name: "test_div".to_string(),
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
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_invalid_opset() {
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Add,
            name: "test_add".to_string(),
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
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        let result = processor.infer_types(&mut node, 6, &prefs);
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
        let processor = ArithmeticBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Add,
            name: "test_add".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }
}
