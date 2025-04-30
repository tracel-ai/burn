use crate::ir::{ArgType, Node};

pub fn squeeze_config(curr: &Node) -> Vec<i64> {
    let axes = curr
        .attrs
        .iter()
        .filter_map(|(key, value)| {
            if key == "axes" {
                Some(value.clone().into_i64s())
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(Vec::new);

    match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    axes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> Node {
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
        if let Some(ref axes_val) = axes {
            attrs.insert("axes".to_string(), AttributeValue::Int64s(axes_val.clone()));
        }

        Node {
            node_type: NodeType::Squeeze,
            name: "test_squeeze".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "squeezed".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: rank - (axes.as_ref().map_or(0, |a| a.len())),
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_squeeze_config_with_axes() {
        let node = create_test_node(Some(vec![0, 2]), 4);
        let axes = squeeze_config(&node);
        assert_eq!(axes, vec![0, 2]);
    }

    #[test]
    fn test_squeeze_config_no_axes() {
        let node = create_test_node(None, 4);
        let axes = squeeze_config(&node);
        assert!(axes.is_empty());
    }
}
