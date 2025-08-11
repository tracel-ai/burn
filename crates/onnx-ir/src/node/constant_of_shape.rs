use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Updates the output rank for a ConstantOfShape node based on the rank of its input.
pub fn constant_of_shape_update_output(node: &mut Node) {
    log::debug!("ConstantOfShape rank inference for node {}", node.name);

    let value_type = node
        .attrs
        .get("value")
        .map(|v| v.clone().into_tensor().elem_type())
        .unwrap_or(ElementType::Float32); // If not given, defaults to 0 as float32
    log::debug!(
        "ConstantOfShape value type for {}: {:?}",
        node.name,
        value_type
    );

    let rank = match &node.inputs[0].ty {
        ArgType::Shape(rank) => {
            log::debug!(
                "ConstantOfShape input is Shape with rank {} for {}",
                rank,
                node.name
            );
            *rank
        }
        ArgType::Tensor(tensor_type) => {
            log::debug!("ConstantOfShape input is Tensor for {}", node.name);
            let r = tensor_type
                .static_shape
                .as_ref()
                .and_then(|shape| shape.first())
                .copied()
                .expect(
                    "ConstantOfShape node must have a Tensor with a non-empty static shape value",
                );
            log::debug!(
                "ConstantOfShape derived rank from tensor: {} for {}",
                r,
                node.name
            );
            r
        }
        _ => panic!("ConstantOfShape node requires a Tensor or Shape type as input"),
    };

    // Update the input type to be a shape
    node.inputs[0].ty = ArgType::Shape(rank);
    log::debug!(
        "ConstantOfShape updated input to Shape({}) for {}",
        rank,
        node.name
    );

    // Optimization: When input is Shape(1) and value type is Int64,
    // output Shape(1) directly instead of a tensor. This is a common pattern
    // in ONNX models where ConstantOfShape is used to create shape arrays.
    // Downstream operations can cast to tensor if needed.
    // This optimization improves performance by keeping shape operations in the Shape domain.
    if rank == 1 && value_type == ElementType::Int64 {
        // Special optimization for Shape(1) with Int64 values
        node.outputs[0].ty = ArgType::Shape(1);
        log::debug!(
            "ConstantOfShape optimization: Shape(1) -> Shape(1) with Int64 value for {}",
            node.name
        );
    } else if rank == 0 {
        // When rank is 0, output should be a scalar
        node.outputs[0].ty = ArgType::Scalar(value_type);
        log::debug!("ConstantOfShape output is Scalar for {}", node.name);
    } else {
        // General case: output is a tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: value_type,
            rank,
            static_shape: None,
        });
        log::debug!("ConstantOfShape output rank for {}: {}", node.name, rank);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, Data, NodeType, TensorData};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(input_ty: ArgType) -> Node {
        NodeBuilder::new(NodeType::ConstantOfShape, "test_constantofshape")
            .add_input("shape", input_ty)
            .output_tensor_f32("output", 0, None) // Will be updated
            .build()
    }

    #[test]
    fn test_shape_input() {
        let mut node = create_test_node(ArgType::Shape(3));

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_tensor_input_with_static_shape() {
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: Some(vec![4]),
        }));

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_custom_value_type() {
        let mut node = create_test_node(ArgType::Shape(2));
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![7]), // Int64 value
            }),
        );

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "ConstantOfShape node requires a Tensor or Shape type as input")]
    fn test_invalid_input_type() {
        let mut node = create_test_node(ArgType::Scalar(ElementType::Float32));
        constant_of_shape_update_output(&mut node);
    }

    #[test]
    fn test_scalar_output_with_shape_0() {
        // Test when input is Shape(0), output should be Scalar
        let mut node = create_test_node(ArgType::Shape(0));

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_tensor_shape_0() {
        // Test when input is a tensor with static shape [0], output should be Scalar
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: Some(vec![0]), // Shape is [0], meaning rank-0 output
        }));

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_custom_value() {
        // Test scalar output with custom value type
        let mut node = create_test_node(ArgType::Shape(0));
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![42]), // Custom Int64 value
            }),
        );

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Int64);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_shape_optimization_with_int64() {
        // Test Shape(1) -> Shape(1) optimization when value type is Int64
        let mut node = create_test_node(ArgType::Shape(1));
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![5]), // Int64 value
            }),
        );

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1);
            }
            _ => panic!("Expected Shape(1) output for Shape(1) input with Int64 value"),
        }
    }

    #[test]
    fn test_shape_1_with_float_no_optimization() {
        // Test that Shape(1) with Float32 does NOT get optimized (outputs Tensor)
        let mut node = create_test_node(ArgType::Shape(1));
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Float32s(vec![1.5]), // Float32 value
            }),
        );

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with Float32 value"),
        }
    }

    #[test]
    fn test_shape_1_default_value_no_optimization() {
        // Test that Shape(1) with default value (Float32) does NOT get optimized
        let mut node = create_test_node(ArgType::Shape(1));
        // No value attribute means default Float32

        constant_of_shape_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with default Float32 value"),
        }
    }
}
