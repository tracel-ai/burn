use crate::ir::{ArgType, Data, Node, TensorType};

/// Updates the output rank and shape for the Expand operation based on the provided shape input.
/// If the shape is a constant, the rank and static shape of the output are set accordingly.
/// If the shape is dynamic, the rank is inferred from the static shape of the shape input.
pub fn expand_update_outputs(node: &mut Node) {
    let shape = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(shape) => Some(shape.clone()),
                _ => panic!("Expand operation encountered invalid input types"),
            },
            None => None,
        }
    } else {
        panic!("Expand operation requires exactly two inputs");
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Expand operation encountered invalid output types"),
    };

    if let Some(shape) = shape {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: shape.len(),
            static_shape: Some(shape.into_iter().map(|dim| dim as usize).collect()),
            ..output
        });
    } else {
        // When the shape cannot be determined statically (i.e., the second argument 'shape' is passed dynamically),
        // infer the rank from the static shape of the input tensor.
        let output_rank = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor
                .static_shape
                .as_ref()
                .expect("Shape input must have a static shape defined")
                .first()
                .copied()
                .expect("Static shape must contain at least one element"),
            ArgType::Shape(rank) => *rank,
            _ => panic!("Shape input must be of tensor or shape type",),
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: output_rank,
            static_shape: None, // The exact shape cannot be determined statically
            ..output
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType, TensorData};
    use std::collections::HashMap;

    fn create_test_node(input_rank: usize, shape_value: Option<Vec<i64>>) -> Node {
        let inputs = vec![
            Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "shape".to_string(),
                ty: if shape_value.is_some() {
                    ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int64,
                        rank: 1,
                        static_shape: Some(vec![shape_value.as_ref().unwrap().len()]),
                    })
                } else {
                    ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int64,
                        rank: 1,
                        static_shape: Some(vec![3]), // Example: a shape with 3 dimensions
                    })
                },
                value: shape_value.map(|shape| TensorData {
                    shape: vec![shape.len()],
                    data: Data::Int64s(shape),
                }),
                passed: true,
            },
        ];

        let outputs = vec![Argument {
            name: "output".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: NodeType::Expand,
            name: "test_expand".to_string(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_expand_with_constant_shape() {
        let mut node = create_test_node(2, Some(vec![2, 3, 4]));

        expand_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![2, 3, 4]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_with_dynamic_shape() {
        let mut node = create_test_node(2, None);

        expand_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, None);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Expand operation requires exactly two inputs")]
    fn test_expand_with_incorrect_inputs() {
        let mut node = create_test_node(2, Some(vec![2, 3, 4]));
        node.inputs.pop(); // Remove one input

        expand_update_outputs(&mut node);
    }
}
