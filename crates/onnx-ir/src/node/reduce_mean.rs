use crate::ir::{ArgType, Node, TensorType};

/// Configuration for ReduceMean operation
pub struct ReduceMeanConfig {
    pub axes: Option<Vec<i64>>,
    pub keepdims: bool,
}

/// Create a ReduceMeanConfig from the attributes of the node
pub fn reduce_mean_config(node: &Node) -> ReduceMeanConfig {
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

    // Note: Multiple axes reduction is now supported

    // Convert negative axes to positive
    let processed_axes = if axes.is_empty() {
        None
    } else {
        let mut processed = Vec::new();
        for mut axis in axes {
            if axis < 0 {
                // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
                axis += tensor.rank as i64;
            }
            processed.push(axis);
        }
        Some(processed)
    };

    ReduceMeanConfig {
        axes: processed_axes,
        keepdims: keepdims != 0,
    }
}

/// Update output rank for ReduceMean based on axes.
pub fn reduce_mean_update_outputs(node: &mut Node) {
    log::debug!("ReduceMean rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ReduceMean: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    let config = reduce_mean_config(node);

    let output_rank = if let Some(ref axes) = config.axes {
        if config.keepdims {
            // Keep the same rank with reduced dimension size = 1
            tensor.rank
        } else {
            // Reduce rank by the number of axes being reduced
            tensor.rank - axes.len()
        }
    } else {
        // Reduce all dimensions - results in scalar (rank 1)
        1
    };

    log::debug!("ReduceMean output rank for {}: {}", node.name, output_rank);

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
        let mut builder = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
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
    fn test_reduce_mean_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, Some(vec![1]));
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_mean_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, Some(vec![1])); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_mean_config_no_axes_keepdims() {
        // When axes is None, it means reduce over all dimensions
        let node = create_test_node(None, Some(1));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, None);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_mean_config_no_axes_no_keepdims() {
        // When axes is None, it means reduce over all dimensions
        let node = create_test_node(None, Some(0));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, None);
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_mean_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, Some(vec![0, 1]));
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_mean_config_multiple_axes_no_keepdims() {
        let node = create_test_node(Some(vec![0, 2]), Some(0));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, Some(vec![0, 2]));
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_mean_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let config = reduce_mean_config(&node);
        assert_eq!(config.axes, Some(vec![1]));
        assert_eq!(config.keepdims, false);
    }
}
