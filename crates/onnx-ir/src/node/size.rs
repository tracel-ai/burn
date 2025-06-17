use crate::ir::{ArgType, ElementType, Node};

/// Update output type for Size (always scalar).
pub fn size_update_outputs(node: &mut Node) {
    log::debug!("Size rank inference for node {}", node.name);

    assert_eq!(
        node.inputs.len(),
        1,
        "Size: expected 1 input, found {}",
        node.inputs.len()
    );

    node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(rank: usize) -> Node {
        let builder = NodeBuilder::new(NodeType::Size, "test_size")
            .input_tensor_f32("data", rank, None)
            .output_scalar_i64("size");

        builder.build()
    }

    #[test]
    fn test_size_update_outputs() {
        let mut node = create_test_node(4);
        size_update_outputs(&mut node);
        assert!(matches!(
            &node.outputs[0].ty,
            ArgType::Scalar(ElementType::Int64)
        ));
    }
}
