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

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: value_type,
        rank,
        static_shape: None,
    });
    log::debug!("ConstantOfShape output rank for {}: {}", node.name, rank);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, Data, NodeType, TensorData};
    use std::collections::HashMap;

    fn create_test_node(input_ty: ArgType) -> Node {
        let inputs = vec![Argument {
            name: "shape".to_string(),
            ty: input_ty,
            value: None,
            passed: true,
        }];

        let outputs = vec![Argument {
            name: "output".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32, // Will be updated
                rank: 0,                         // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let attrs = HashMap::new();
        // Default value attribute not set initially

        Node {
            node_type: NodeType::ConstantOfShape,
            name: "test_constantofshape".to_string(),
            inputs,
            outputs,
            attrs,
        }
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
}
