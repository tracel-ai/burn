//! Processor for basic arithmetic operations (Add, Sub, Mul, Div)
//!
//! These operations perform element-wise arithmetic with broadcasting support.
//! This processor includes special handling for type propagation when performing
//! arithmetic with constants on Shape or Scalar types.

use crate::ir::Node;
use crate::processor::NodeProcessor;
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
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        // Arithmetic binary operations require exactly two inputs
        assert_eq!(node.inputs.len(), 2);

        // Type propagation for Shape arithmetic:
        // When performing arithmetic on a Shape with a constant, convert the constant to Shape type.
        // This is common in dynamic shape calculations like:
        // - new_shape = old_shape + offset
        // - half_shape = old_shape / 2
        // - scaled_shape = old_shape * factor

        // Case 1: Shape op Constant => Shape op Shape
        if node.inputs[0].ty.is_shape() && node.inputs[1].has_value() {
            node.inputs[1].should_be(node.inputs[0].ty.clone());
        }

        // Case 2: Constant op Shape => Shape op Shape
        if node.inputs[1].ty.is_shape() && node.inputs[0].has_value() {
            node.inputs[0].should_be(node.inputs[1].ty.clone());
        }

        // Type propagation for Scalar arithmetic:
        // When performing arithmetic on a Scalar with a constant, convert the constant to Scalar type.
        // This preserves scalar semantics through arithmetic operations.

        // Case 3: Scalar op Constant => Scalar op Scalar
        if node.inputs[0].ty.is_scalar() && node.inputs[1].has_value() {
            node.inputs[1].should_be(node.inputs[0].ty.clone());
        }

        // Case 4: Constant op Scalar => Scalar op Scalar
        if node.inputs[1].ty.is_scalar() && node.inputs[0].has_value() {
            node.inputs[0].should_be(node.inputs[1].ty.clone());
        }

        // Apply standard broadcasting rules to infer output type
        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_arithmetic_add() {
        let processor = ArithmeticBinaryProcessor;

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
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_sub() {
        let processor = ArithmeticBinaryProcessor;

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
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_mul() {
        let processor = ArithmeticBinaryProcessor;

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
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 4),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_arithmetic_div() {
        let processor = ArithmeticBinaryProcessor;

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
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }
}
