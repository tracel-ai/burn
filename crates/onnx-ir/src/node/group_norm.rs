use crate::ir::Node;

/// Configuration for GroupNorm operations
#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Number of groups
    pub num_groups: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

impl GroupNormConfig {
    /// Create a new GroupNormConfig
    pub fn new(num_features: usize, num_groups: usize, epsilon: f64) -> Self {
        Self {
            num_features,
            num_groups,
            epsilon,
        }
    }
}

/// Create a GroupNormConfig from the attributes of the node
pub fn group_norm_config(node: &Node) -> (GroupNormConfig, bool) {
    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("GroupNorm: weight tensor must be present")
        .shape
        .clone();

    let mut stash_type = 1;
    let num_features = weight_shape[0];
    let mut num_groups = None;
    let mut epsilon = 1e-5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "epsilon" => epsilon = value.clone().into_f32(),
            "num_groups" => num_groups = Some(value.clone().into_i64() as usize),
            "stash_type" => stash_type = value.clone().into_i64(),
            _ => panic!("Unexpected attribute for GroupNorm: {key}"),
        }
    }

    let num_groups = num_groups.expect("GroupNorm: num_groups attribute must be present");
    if num_groups > 0 && !num_features.is_multiple_of(num_groups) {
        panic!("GroupNorm: number of features must be divisible by the number of groups");
    }

    (
        GroupNormConfig::new(num_features, num_groups, epsilon as f64),
        stash_type == 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        epsilon: f32,
        num_features: usize,
        num_groups: usize,
        stash_type: i64,
    ) -> Node {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::GroupNormalization, "test_groupnorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_int("num_groups", num_groups as i64)
            .attr_int("stash_type", stash_type)
            .attr_float("epsilon", epsilon)
            .build()
    }

    #[test]
    fn test_group_norm_config_basic() {
        let node = create_test_node(1e-5, 64, 8, 1);
        let (config, full_precision) = group_norm_config(&node);

        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(full_precision);
    }

    #[test]
    fn test_group_norm_config_no_stash_type() {
        let node = create_test_node(1e-5, 64, 8, 0);
        let (config, full_precision) = group_norm_config(&node);

        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(!full_precision);
    }

    #[test]
    #[should_panic]
    fn test_group_norm_config_invalid_num_groups() {
        // num features is not divisible by num groups
        let node = create_test_node(1e-5, 64, 7, 0);
        let _ = group_norm_config(&node);
    }
}
