use crate::{
    Argument, ElementType, TensorData,
    ir::{ArgType, Data, Node, TensorType},
};

/// Updates the output rank and shape for the Expand operation based on the provided shape input.
/// If the shape is a constant, the rank and static shape of the output are set accordingly.
/// If the shape is dynamic, the rank is inferred from the static shape of the shape input.
pub fn expand_update_outputs(node: &mut Node) {
    log::debug!("Expand node {} has {} inputs", node.name, node.inputs.len());
    if node.inputs.len() >= 2 {
        log::debug!(
            "Expand node {} input[0]: {:?}",
            node.name,
            node.inputs[0].ty
        );
        log::debug!(
            "Expand node {} input[1]: {:?}",
            node.name,
            node.inputs[1].ty
        );
    }

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

    // Get input element type - Expand should preserve the input's element type
    let input_elem_type = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        _ => panic!("Expand operation requires first input to be a tensor"),
    };

    let output = match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.clone(),
        _ => panic!("Expand operation encountered invalid output types"),
    };

    if let Some(shape) = shape {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: input_elem_type.clone(),
            rank: shape.len(),
            static_shape: Some(shape.into_iter().map(|dim| dim as usize).collect()),
        });
    } else {
        // When the shape cannot be determined statically (i.e., the second argument 'shape' is passed dynamically),
        // infer the rank from the shape input.
        let output_rank = match &node.inputs[1].ty {
            ArgType::Shape(rank) => {
                // Shape type directly gives us the output rank
                *rank
            }
            ArgType::Tensor(tensor) => {
                // For tensor inputs representing shapes, the rank should be 1
                // and the output rank is determined by the number of elements
                if tensor.rank == 1 {
                    // If we have a static shape, use it to get the exact output rank
                    if let Some(static_shape) = &tensor.static_shape {
                        static_shape
                            .first()
                            .copied()
                            .expect("Static shape must contain at least one element")
                    } else {
                        // For dynamic rank-1 tensors without static shape, we need to make an assumption
                        // or get the information from elsewhere.
                        // Check if we have a value that can tell us the rank
                        if let Some(value) = &node.inputs[1].value {
                            if let Data::Int64s(shape_data) = &value.data {
                                // We have the actual shape values, so the output rank is the number of elements
                                shape_data.len()
                            } else {
                                panic!(
                                    "Expand shape tensor has unexpected data type: {:?}",
                                    value.data
                                )
                            }
                        } else {
                            // No static shape and no value - this is truly dynamic
                            // We need to look at the output type if it's been set
                            log::warn!(
                                "Expand node {} has dynamic shape tensor without static shape info. Using output rank if available.",
                                node.name
                            );
                            // Use the current output rank if it's already a tensor
                            match &output {
                                TensorType { rank, .. } if *rank > 0 => *rank,
                                _ => panic!(
                                    "Cannot determine output rank for Expand node {} with fully dynamic shape tensor. Please provide static shape or use Shape type.",
                                    node.name
                                ),
                            }
                        }
                    }
                } else {
                    panic!(
                        "Shape tensor for Expand must be 1-dimensional, got rank {}",
                        tensor.rank
                    )
                }
            }
            _ => panic!("Shape input must be of tensor or shape type"),
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: input_elem_type,
            rank: output_rank,
            static_shape: None, // The exact shape cannot be determined statically
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
    use crate::ir::{ElementType, NodeType, TensorData};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        input_rank: usize,
        shape_value: Option<Vec<i64>>,
        shape_type: Option<ArgType>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", input_rank, None)
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(shape) = shape_value {
            builder = builder.input_tensor_i64_data("shape", shape.clone(), vec![shape.len()]);
        } else if let Some(st) = shape_type {
            // Use the provided custom shape type
            builder = builder.add_input("shape", st);
        } else {
            // Default case with dynamic shape
            builder = builder.input_tensor_i64("shape", 1, Some(vec![3]));
        }

        builder.build()
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

    #[test]
    fn test_expand_update_outputs_with_shape_input() {
        // Test Expand with Shape type as shape input
        let mut node = create_test_node(2, None, Some(ArgType::Shape(4)));

        expand_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // Shape(4) means output will be rank 4
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_update_outputs_with_shape_input_static_value() {
        // Test Expand with Shape type that has a static value
        let mut node = create_test_node(2, None, Some(ArgType::Shape(3)));

        // Add a static value to the shape input
        node.inputs[1].value = Some(TensorData {
            shape: vec![3],
            data: Data::Int64s(vec![5, 10, 15]),
        });

        expand_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![5, 10, 15]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_preserves_input_element_type() {
        // Test that Expand preserves the input element type for different types

        // Test Float32 -> Float32
        {
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_f32("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_f32("output", 0, None)
                .build();

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64, // Wrong type
                rank: 0,
                static_shape: None,
            });

            expand_update_outputs(&mut node);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Float32,
                        "Expand should preserve Float32 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Int64 -> Int64
        {
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_i64("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_i64("output", 0, None)
                .build();

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            expand_update_outputs(&mut node);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Int64,
                        "Expand should preserve Int64 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Bool -> Bool
        {
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_bool("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_bool("output", 0, None)
                .build();

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            expand_update_outputs(&mut node);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Bool,
                        "Expand should preserve Bool input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    fn test_expand_with_mismatched_output_type() {
        // Test that Expand corrects output type even when initially set incorrectly
        // This simulates the case where ONNX might have wrong type info
        let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_i64("input", 2, None) // Input is Int64
            .input_tensor_i64_data("shape", vec![2, 3], vec![2])
            .output_tensor_f32("output", 0, None) // Output incorrectly set to Float32
            .build();

        expand_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.elem_type,
                    ElementType::Int64,
                    "Expand should use input type (Int64) not initial output type (Float32)"
                );
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![2, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
