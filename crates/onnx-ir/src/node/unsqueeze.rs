use crate::ir::{ArgType, Data, Node, TensorType};

/// Update output rank for Unsqueeze based on axes.
/// Update the output tensor dimension based on the "axes" attribute or the second input
pub fn unsqueeze_update_output(node: &mut Node) {
    log::debug!("Unsqueeze rank inference for node {}", node.name);

    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(a) => Some(a.clone()),
                _ => panic!("Unsqueeze: invalid input types"),
            },
            None => None,
        }
    } else {
        let axes = node.attrs.get("axes").cloned().map(|v| {
            let axes = v.into_i64s();
            log::debug!(
                "Unsqueeze axes from attribute for {}: {:?}",
                node.name,
                axes
            );
            axes
        });
        axes
    };

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => {
            0 // treat scalar as 0-dim tensor
        }
        _ => panic!("Unsqueeze: invalid input type"),
    };

    let output_elem = match &node.outputs[0].ty {
        ArgType::Tensor(_) => node.inputs[0].ty.elem_type().clone(),
        ArgType::Scalar(elem_type) => elem_type.clone(),
        _ => panic!("Unsqueeze: invalid output type"),
    };

    let output_rank = if let Some(axes) = axes {
        input_rank + axes.len()
    } else if let ArgType::Tensor(tensor) = &node.inputs[1].ty {
        if let Some(static_shape) = &tensor.static_shape {
            input_rank + *static_shape.first().expect("Empty shape")
        } else {
            panic!("Unsqueeze: should have static shape")
        }
    } else {
        panic!("Unsqueeze: missing axes information")
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape: None, // shape is tracked and calculated at runtime
        elem_type: output_elem,
    });

    log::debug!("Unsqueeze output rank for {}: {}", node.name, output_rank);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorData};
    use std::collections::HashMap;

    fn create_test_node_with_attr(input_rank: usize, axes: Vec<i64>) -> Node {
        let inputs = vec![Argument {
            name: "X".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: input_rank,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let outputs = vec![Argument {
            name: "Y".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), AttributeValue::Int64s(axes.clone()));

        Node {
            node_type: NodeType::Unsqueeze,
            name: "test_unsqueeze".to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    fn create_test_node_with_input(input_rank: usize, axes: Vec<i64>) -> Node {
        let inputs = vec![
            Argument {
                name: "X".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "axes".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: Some(vec![axes.len()]),
                }),
                value: Some(TensorData {
                    data: Data::Int64s(axes),
                    shape: vec![1],
                }),
                passed: true,
            },
        ];

        let outputs = vec![Argument {
            name: "Y".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: NodeType::Unsqueeze,
            name: "test_unsqueeze".to_string(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_unsqueeze_with_attr() {
        let mut node = create_test_node_with_attr(2, vec![0, 3]);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // 2 + 2 = 4
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_with_input() {
        let mut node = create_test_node_with_input(3, vec![1, 2, 4]);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 6); // 3 + 3 = 6
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unsqueeze_scalar() {
        let mut node = create_test_node_with_attr(0, vec![0]);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        unsqueeze_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1); // 0 + 1 = 1
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Unsqueeze: invalid input type")]
    fn test_unsqueeze_invalid_input() {
        let mut node = create_test_node_with_attr(2, vec![0]);
        node.inputs[0].ty = ArgType::Shape(1);
        unsqueeze_update_output(&mut node);
    }
}
