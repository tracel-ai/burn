use crate::ir::{ArgType, Node};

/// Create a ReduceSumConfig from the attributes of the node
pub fn reduce_sum_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "keepdims" => keepdims = value.clone().into_i64(),
            "axes" => axes = value.clone().into_i64s(),
            // TODO: handle noop_with_empty_axes
            _ => {}
        }
    }

    // Process axes from additional input (if available)
    if let Some(value) = node
        .inputs
        .get(1)
        .and_then(|argument| argument.value.as_ref())
    {
        axes = value.clone().data.into_i64s();
    }

    if axes.len() > 1 {
        panic!("ReduceSum: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceSum: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        // Not supported in Burn
        panic!("ReduceSum: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
            dim += tensor.rank as i64;
        }
        Some(dim as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(
        axes: Option<Vec<i64>>,
        keepdims: Option<i64>,
        with_axes_input: bool,
    ) -> Node {
        let mut inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        // Add axes input if requested
        if with_axes_input && axes.is_some() {
            let axes_clone = axes.clone().unwrap();
            inputs.push(Argument {
                name: "axes".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(TensorData {
                    data: Data::Int64s(axes_clone.clone()),
                    shape: vec![axes_clone.len()],
                }),
                passed: true,
            });
        }

        let mut attrs = HashMap::new();
        if !with_axes_input && axes.is_some() {
            attrs.insert(
                "axes".to_string(),
                AttributeValue::Int64s(axes.clone().unwrap()),
            );
        }
        if let Some(kd) = keepdims {
            attrs.insert("keepdims".to_string(), AttributeValue::Int64(kd));
        }

        Node {
            node_type: NodeType::ReduceSum,
            name: "test_reduce_sum".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "reduced".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_reduce_sum_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1), false);
        let dim = reduce_sum_config(&node);
        assert_eq!(dim, Some(1));
    }

    #[test]
    fn test_reduce_sum_config_with_input_axes() {
        let node = create_test_node(Some(vec![1]), Some(1), true);
        let dim = reduce_sum_config(&node);
        assert_eq!(dim, Some(1));
    }

    #[test]
    fn test_reduce_sum_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1), false);
        let dim = reduce_sum_config(&node);
        assert_eq!(dim, Some(1)); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "ReduceSum: axes must be provided with keepdims")]
    fn test_reduce_sum_config_no_axes() {
        let node = create_test_node(None, Some(1), false);
        let _ = reduce_sum_config(&node);
    }

    #[test]
    #[should_panic(expected = "ReduceSum: reducing on multiple dimensions is not supported")]
    fn test_reduce_sum_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1), false);
        let _ = reduce_sum_config(&node);
    }

    #[test]
    #[should_panic(
        expected = "ReduceSum: the reduce operation must preserve the reduced dimension"
    )]
    fn test_reduce_sum_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0), false);
        let _ = reduce_sum_config(&node);
    }
}
