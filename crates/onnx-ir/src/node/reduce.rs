use crate::{ArgType, Node, TensorType};

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub dims: Vec<usize>,
    pub keepdims: bool,
}

impl ReduceConfig {
    pub fn new(dims: Vec<usize>, keepdims: bool) -> Self {
        Self { dims, keepdims }
    }
}

/// Create a Reduce config from the attributes of the node
pub fn reduce_config(node: &Node) -> ReduceConfig {
    let mut axes = Vec::new();
    let mut keepdims = 1;

    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("{}: Only tensor input is valid", node.node_type),
    };

    // Extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axes" => axes = value.clone().into_i64s(),
            "keepdims" => keepdims = value.clone().into_i64(),
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

    let mut dims: Vec<usize> = axes
        .into_iter()
        .map(|mut dim| {
            if dim < 0 {
                // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
                dim += tensor.rank as i64;
            }
            dim as usize
        })
        .collect();

    // Sort the dimensions to ensure consistent order
    dims.sort();

    ReduceConfig::new(dims, keepdims == 1)
}

/// Update output rank for Reduce based on config.
pub fn reduce_update_outputs(node: &mut Node) {
    log::debug!("{} rank inference for node {}", node.node_type, node.name);

    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("{}: Only tensor input is valid", node.node_type),
    };
    log::debug!(
        "{} input rank for {}: {}",
        node.node_type,
        node.name,
        tensor.rank
    );

    let config = reduce_config(node);

    // Determine if the output should be a scalar
    let should_be_scalar =
        !config.keepdims && (config.dims.is_empty() || config.dims.len() == tensor.rank);

    if should_be_scalar {
        // Output is a scalar
        log::debug!("{} output is scalar for node {}", node.node_type, node.name);
        node.outputs[0].ty = ArgType::Scalar(tensor.elem_type.clone());
    } else {
        // Output is a tensor
        let output_rank = if config.keepdims {
            tensor.rank
        } else {
            tensor.rank - config.dims.len()
        };

        // Infer static shape based if given
        let output_shape = tensor.static_shape.clone().map(|mut shape| {
            if config.keepdims {
                for dim in config.dims {
                    shape[dim] = 1;
                }
                shape
            } else {
                for dim in config.dims.iter().rev() {
                    shape.remove(*dim);
                }
                shape
            }
        });

        log::debug!(
            "{} output rank for {}: {}",
            node.node_type,
            node.name,
            output_rank
        );

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: output_rank,
            static_shape: output_shape,
        });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::ReduceMax, "test_reduce_max")
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
    fn test_reduce_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let config = reduce_config(&node);

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let config = reduce_config(&node);

        assert_eq!(config.dims, [1]); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let config = reduce_config(&node);

        assert_eq!(config.dims, []);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let config = reduce_config(&node);

        assert_eq!(config.dims, [0, 1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let config = reduce_config(&node);

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_update_outputs_scalar_no_axes_no_keepdims() {
        // Test that reduce with no axes and keepdims=false produces a scalar output
        let mut node = create_test_node(None, Some(0));
        reduce_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_scalar_all_dims_no_keepdims() {
        // Test that reduce with all dimensions and keepdims=false produces a scalar output
        let mut node = create_test_node(Some(vec![0, 1, 2]), Some(0));
        reduce_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_partial_dims_no_keepdims() {
        // Test that reduce with partial dimensions and keepdims=false produces a tensor output
        let mut node = create_test_node(Some(vec![1]), Some(0));
        reduce_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2)
                assert_eq!(tensor.rank, 2);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_with_keepdims() {
        // Test that reduce with keepdims=true always produces a tensor output
        let mut node = create_test_node(None, Some(1));
        reduce_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain original rank when keepdims=true
                assert_eq!(tensor.rank, 3);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar when keepdims=true");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }
}
