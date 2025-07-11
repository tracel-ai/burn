use crate::ir::Node;

/// Create a HardSigmoidConfig from the alpha and beta attributes of the node
pub fn hard_sigmoid_config(node: &Node) -> (f64, f64) {
    let mut alpha = 0.2;
    let mut beta = 0.5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "alpha" => alpha = value.clone().into_f32() as f64,
            "beta" => beta = value.clone().into_f32() as f64,
            _ => {}
        }
    }

    (alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(alpha: f32, beta: f32) -> Node {
        NodeBuilder::new(NodeType::HardSigmoid, "test_hard_sigmoid")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .attr_float("beta", beta)
            .build()
    }

    #[test]
    fn test_hard_sigmoid_config_with_attrs() {
        let node = create_test_node(0.3, 0.6);
        let (alpha, beta) = hard_sigmoid_config(&node);
        assert!((alpha - 0.3).abs() < 1e-6);
        assert!((beta - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_config_default() {
        let mut node = create_test_node(0.3, 0.6);
        node.attrs.clear(); // Remove all attributes
        let (alpha, beta) = hard_sigmoid_config(&node);
        assert_eq!(alpha, 0.2); // Check default values
        assert_eq!(beta, 0.5);
    }
}
