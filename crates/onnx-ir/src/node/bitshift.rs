use crate::ir::Node;

/// Configuration for BitShift operation
pub fn bitshift_config(node: &Node) -> String {
    node.attrs
        .get("direction")
        .map(|val| val.clone().into_string())
        .unwrap_or("left".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_bitshift_config_with_direction_left() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "left")
            .build();

        let config = bitshift_config(&node);
        assert_eq!(config, "left");
    }

    #[test]
    fn test_bitshift_config_with_direction_right() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "right")
            .build();

        let config = bitshift_config(&node);
        assert_eq!(config, "right");
    }

    #[test]
    fn test_bitshift_config_default_direction() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .build();

        let config = bitshift_config(&node);
        assert_eq!(config, "left");
    }
}