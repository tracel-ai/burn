use crate::ir::Node;

/// Create a LeakyReluConfig from the alpha attribute of the node
pub fn leaky_relu_config(node: &Node) -> f64 {
    let mut alpha = 0.01;

    for (key, value) in node.attrs.iter() {
        if key.as_str() == "alpha" {
            alpha = value.clone().into_f32() as f64
        }
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(alpha: f32) -> Node {
        NodeBuilder::new(NodeType::LeakyRelu, "test_leaky_relu")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_leaky_relu_config_with_alpha() {
        let node = create_test_node(0.2);
        let alpha = leaky_relu_config(&node);
        assert!((alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let alpha = leaky_relu_config(&node);
        assert_eq!(alpha, 0.01); // Check default value
    }
}
