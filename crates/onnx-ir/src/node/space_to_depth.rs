use crate::{ArgType, TensorType, ir::Node};

/// Update output type for SpaceToDepth operation (rank 4).
pub fn space_to_depth_update_outputs(node: &mut Node) {
    log::debug!("SpaceToDepth rank inference for node {}", &node.name);

    // Extract the input tensor type to determine rank and shape
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("SpaceToDepth: only tensor input is valid"),
    };
    assert_eq!(
        tensor.rank, 4,
        "SpaceToDepth: only rank 4 tensors are supported"
    );

    // Get the block size from attribute
    let block_size = node
        .attrs
        .get("blocksize")
        .cloned()
        .expect("SpaceToDepth: blocksize attribute not found")
        .into_i64() as usize;

    log::debug!(
        "SpaceToDepth blocksize from attribute for {}: {:?}",
        &node.name,
        block_size
    );

    // Infer static shape based on rank and block size
    let static_shape = tensor.static_shape.clone().map(|shape| {
        let [b, c, h, w] = shape
            .try_into()
            .expect("SpaceToDepth: input tensor rank is not 4");
        vec![
            b,
            c * block_size * block_size,
            h / block_size,
            w / block_size,
        ]
    });

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: tensor.rank,
        static_shape,
    });
}

/// Get the configuration from the attributes of the node
pub fn space_to_depth_config(node: &Node) -> usize {
    let mut block_size: Option<usize> = None;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "blocksize" => block_size = Some(value.clone().into_i64() as usize),
            _ => panic!("Unexpected attribute for SpaceToDepth: {key}"),
        }
    }

    let block_size = block_size.expect("SpaceToDepth: blocksize must be provided");
    assert!(
        block_size > 0,
        "SpaceToDepth: block_size must be greater than 0"
    );

    block_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(rank: usize, static_shape: Option<Vec<usize>>, block_size: i64) -> Node {
        let builder = NodeBuilder::new(NodeType::DepthToSpace, "test_space_to_depth")
            .input_tensor_f32("input", rank, static_shape)
            .output_tensor_f32("output", rank, None) // Same rank as input
            .attr_int("blocksize", block_size);
        builder.build()
    }

    #[test]
    fn test_basic_config() {
        let node = create_test_node(4, None, 2);
        let block_size = space_to_depth_config(&node);

        assert_eq!(block_size, 2);
    }

    #[test]
    fn test_static_shape_update_outputs() {
        let mut node = create_test_node(4, Some(vec![2, 1, 4, 6]), 2);
        space_to_depth_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, vec![2, 4, 2, 3].into());
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
