use crate::{
    Argument, ElementType, TensorData,
    ir::{ArgType, Data, Node, TensorType},
};

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

/// Shape information for the Expand operation.
#[derive(Debug, Clone)]
pub enum ExpandShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape that will be determined during execution.
    Runtime(Argument),
}

/// Creates an ExpandShape configuration from the given Node.
///
/// Extracts shape information from the node's second input to determine
/// whether to use static or runtime shape expansion.
pub fn expand_config(node: &Node) -> ExpandShape {
    match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => {
            assert_eq!(tensor.rank, 1, "Expand: shape tensor must be 1D");
            assert!(
                matches!(tensor.elem_type, ElementType::Int64),
                "Expand: shape tensor must have element type int64"
            );
        }
        ArgType::Shape(_) => {
            // Shapes are always 1-D int64 data, so nothing to assert here
        }
        _ => panic!("Only tensor input is valid for shape"),
    }

    match &node.inputs[1].value {
        Some(TensorData {
            data: Data::Int64s(shape),
            ..
        }) => ExpandShape::Static(shape.clone()),
        None => {
            // we were unable to statically determine the input value, so we'll need to fetch it at runtime
            ExpandShape::Runtime(node.inputs[1].clone())
        }
        _ => panic!(
            "Shape data type must be int64, is {:?}",
            &node.inputs[1].value
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType, TensorData};
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        shape_value: Option<Vec<i64>>,
        shape_type: Option<ArgType>,
    ) -> Node {
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
                ty: shape_type.unwrap_or_else(|| {
                    if shape_value.is_some() {
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
                    }
                }),
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
        let mut node = create_test_node(2, Some(vec![2, 3, 4]), None);

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
        let mut node = create_test_node(2, None, None);

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
        let mut node = create_test_node(2, Some(vec![2, 3, 4]), None);
        node.inputs.pop(); // Remove one input

        expand_update_outputs(&mut node);
    }

    // Tests for expand_config function

    #[test]
    fn test_expand_config_with_static_shape() {
        let node = create_test_node(2, Some(vec![2, 3, 4]), None);
        let config = expand_config(&node);

        match config {
            ExpandShape::Static(shape) => {
                assert_eq!(shape, vec![2, 3, 4]);
            }
            ExpandShape::Runtime(_) => panic!("Expected Static config, got Runtime"),
        }
    }

    #[test]
    fn test_expand_config_with_runtime_shape() {
        let node = create_test_node(2, None, None);
        let config = expand_config(&node);

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(arg) => {
                assert_eq!(arg.name, "shape");
                match arg.ty {
                    ArgType::Tensor(tensor) => {
                        assert_eq!(tensor.elem_type, ElementType::Int64);
                        assert_eq!(tensor.rank, 1);
                    }
                    _ => panic!("Expected tensor type for runtime shape"),
                }
            }
        }
    }

    #[test]
    fn test_expand_config_with_shape_type() {
        let shape_type = ArgType::Shape(3);
        let node = create_test_node(2, None, Some(shape_type));
        let config = expand_config(&node);

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(arg) => {
                assert_eq!(arg.name, "shape");
                match arg.ty {
                    ArgType::Shape(rank) => {
                        assert_eq!(rank, 3);
                    }
                    _ => panic!("Expected shape type for runtime shape"),
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Expand: shape tensor must be 1D")]
    fn test_expand_config_with_invalid_shape_rank() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 2, // Invalid rank, should be 1
            static_shape: None,
        });
        let node = create_test_node(2, None, Some(invalid_shape_type));
        let _ = expand_config(&node);
    }

    #[test]
    #[should_panic(expected = "Expand: shape tensor must have element type int64")]
    fn test_expand_config_with_invalid_shape_type() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            elem_type: ElementType::Float32, // Invalid element type, should be Int64
            rank: 1,
            static_shape: None,
        });
        let node = create_test_node(2, None, Some(invalid_shape_type));
        let _ = expand_config(&node);
    }

    #[test]
    #[should_panic(expected = "Only tensor input is valid for shape")]
    fn test_expand_config_with_invalid_input_type() {
        let invalid_shape_type = ArgType::Scalar(ElementType::Int64);
        let node = create_test_node(2, None, Some(invalid_shape_type));
        let _ = expand_config(&node);
    }

    #[test]
    #[should_panic(expected = "Shape data type must be int64")]
    fn test_expand_config_with_invalid_value_type() {
        let mut node = create_test_node(2, None, None);

        // Replace the value with a non-Int64s value
        node.inputs[1].value = Some(TensorData {
            shape: vec![1],
            data: Data::Float32s(vec![1.0]), // Invalid data type
        });

        let _ = expand_config(&node);
    }
}
