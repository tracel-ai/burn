use crate::ir::{ArgType, AttributeValue, Node, TensorType};

/// Create a ReduceMinConfig from the attributes of the node
pub fn reduce_min_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            _ => {}
        }
    }

    if axes.len() > 1 {
        panic!("ReduceMin: reducing on multiple dimensions is not supported")
    }

    if axes.is_empty() && keepdims == 1 {
        panic!("ReduceMin: axes must be provided with keepdims")
    }

    if !axes.is_empty() && keepdims == 0 {
        panic!("ReduceMin: the reduce operation must preserve the reduced dimension")
    }

    if axes.is_empty() {
        None
    } else {
        let mut dim = axes[0];

        if dim < 0 {
            dim += tensor.rank as i64;
        }
        Some(dim as usize)
    }
}

/// Update output rank for ReduceMin based on axes.
pub fn reduce_min_update_outputs(node: &mut Node) {
    log::debug!("ReduceMin rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ReduceMin: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceMin input rank for {}: {}", node.name, tensor.rank);

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceMin output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> Node {
        let inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        if let Some(axes_val) = axes {
            attrs.insert("axes".to_string(), AttributeValue::Int64s(axes_val.clone()));
        }
        if let Some(kd) = keepdims {
            attrs.insert("keepdims".to_string(), AttributeValue::Int64(kd));
        }

        Node {
            node_type: NodeType::ReduceMin,
            name: "test_reduce_min".to_string(),
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
    fn test_reduce_min_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let dim = reduce_min_config(&node);
        assert_eq!(dim, Some(1));
    }

    #[test]
    fn test_reduce_min_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let dim = reduce_min_config(&node);
        assert_eq!(dim, Some(1)); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "ReduceMin: axes must be provided with keepdims")]
    fn test_reduce_min_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let _ = reduce_min_config(&node);
    }

    #[test]
    #[should_panic(expected = "ReduceMin: reducing on multiple dimensions is not supported")]
    fn test_reduce_min_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let _ = reduce_min_config(&node);
    }

    #[test]
    #[should_panic(
        expected = "ReduceMin: the reduce operation must preserve the reduced dimension"
    )]
    fn test_reduce_min_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let _ = reduce_min_config(&node);
    }
}
