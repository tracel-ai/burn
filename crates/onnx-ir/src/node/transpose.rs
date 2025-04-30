use crate::ir::{ArgType, Node};

pub fn transpose_config(curr: &Node) -> Vec<i64> {
    if curr.inputs.len() != 1 {
        panic!(
            "Transpose: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: reverse the dimensions
    let mut perm = (0..tensor.rank as i64).rev().collect::<Vec<i64>>();

    if let Some(axes) = curr.attrs.get("perm") {
        perm = axes.clone().into_i64s();
    }

    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(perm: Option<Vec<i64>>, rank: usize) -> Node {
        let inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        if let Some(perm_val) = perm {
            attrs.insert("perm".to_string(), AttributeValue::Int64s(perm_val));
        }

        Node {
            node_type: NodeType::Transpose,
            name: "test_transpose".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "transposed".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_transpose_config_default() {
        let node = create_test_node(None, 3);
        let perm = transpose_config(&node);
        assert_eq!(perm, vec![2, 1, 0]); // Default is to reverse the dimensions
    }

    #[test]
    fn test_transpose_config_with_perm() {
        let node = create_test_node(Some(vec![0, 2, 1]), 3);
        let perm = transpose_config(&node);
        assert_eq!(perm, vec![0, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "Transpose: multiple inputs are not supported")]
    fn test_transpose_config_multiple_inputs() {
        let mut node = create_test_node(None, 3);
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        });
        let _ = transpose_config(&node);
    }
}
