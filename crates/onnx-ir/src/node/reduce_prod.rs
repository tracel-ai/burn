use crate::ir::{ArgType, AttributeValue, Node, TensorType};

/// Create a `ReduceProdConfig` from the attributes of the node
#[must_use]
pub fn reduce_prod_config(node: &Node) -> Option<usize> {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Extract the attributes
    for (key, value) in &node.attrs {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
            // TODO: handle noop_with_empty_axes (opset 18)
            _ => {}
        }
    }

    assert!(
        (axes.len() <= 1),
        "ReduceProd: reducing on multiple dimensions is not supported"
    );

    assert!(
        !(axes.is_empty() && keepdims == 1),
        "ReduceProd: axes must be provided with keepdims"
    );

    // Not supported in Burn
    assert!(
        !(!axes.is_empty() && keepdims == 0),
        "ReduceProd: the reduce operation must preserve the reduced dimension"
    );

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

/// Update output rank for `ReduceProd` based on axes.
pub fn reduce_prod_update_outputs(node: &mut Node) {
    log::debug!("ReduceProd rank inference for node {}", node.name);

    assert!(
        (node.inputs.len() == 1),
        "ReduceProd: multiple inputs are not supported"
    );
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };
    log::debug!("ReduceProd input rank for {}: {}", node.name, tensor.rank);

    let dim_only = match node.attrs.get("axes") {
        Some(value) => match &value {
            AttributeValue::Int64(_) => true,
            AttributeValue::Int64s(ints) => ints.len() == 1,
            _ => false,
        },
        None => false,
    };

    let output_rank = if dim_only { tensor.rank } else { 1 };
    log::debug!("ReduceProd output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: output_rank,
        static_shape: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::ReduceProd, "test_reduce_prod")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("reduced", 3, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }
        if let Some(kd) = keepdims {
            builder = builder.attr_int("keepdims", kd);
        }

        builder.build()
    }

    #[test]
    fn test_reduce_prod_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let dim = reduce_prod_config(&node);
        assert_eq!(dim, Some(1));
    }

    #[test]
    fn test_reduce_prod_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let dim = reduce_prod_config(&node);
        assert_eq!(dim, Some(1)); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "ReduceProd: axes must be provided with keepdims")]
    fn test_reduce_prod_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let _ = reduce_prod_config(&node);
    }

    #[test]
    #[should_panic(expected = "ReduceProd: reducing on multiple dimensions is not supported")]
    fn test_reduce_prod_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let _ = reduce_prod_config(&node);
    }

    #[test]
    #[should_panic(
        expected = "ReduceProd: the reduce operation must preserve the reduced dimension"
    )]
    fn test_reduce_prod_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let _ = reduce_prod_config(&node);
    }
}
