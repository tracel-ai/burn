use crate::ir::{ArgType, Data, Node, TensorType};

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

/// Update output rank for Squeeze based on axes.
pub fn squeeze_update_output(node: &mut Node) {
    log::debug!("Squeeze rank inference for node {}", node.name);

    let axes = if node.inputs.len() == 2 {
        match &node.inputs[1].value {
            Some(value) => match &value.data {
                Data::Int64s(axes) => Some(axes.clone()),
                _ => panic!("Squeeze: invalid input types"),
            },
            None => None,
        }
    } else {
        node.attrs.get("axes").cloned().map(|v| v.into_i64s())
    };

    let axes = axes.unwrap_or_else(|| panic!("Squeeze must specify an axis"));
    log::debug!("Squeeze axes for {}: {:?}", node.name, axes);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ty => panic!("Squeeze: invalid input type: {ty:?}"),
    };

    log::debug!("Squeeze input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank - axes.len();
    log::debug!("Squeeze output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> Node {
        let output_rank = rank - (axes.as_ref().map_or(0, |a| a.len()));

        let mut builder = NodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("squeezed", output_rank, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }

        builder.build()
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
