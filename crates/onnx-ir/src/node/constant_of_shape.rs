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

    // When rank is 0, output should be a scalar
    if rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(value_type);
        log::debug!("ConstantOfShape output is Scalar for {}", node.name);
    } else {
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
}
