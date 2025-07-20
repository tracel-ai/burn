use crate::Node;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsInfConfig {
    pub detect_negative: bool,
    pub detect_positive: bool,
}

impl IsInfConfig {
    pub fn new(detect_negative: bool, detect_positive: bool) -> Self {
        Self {
            detect_negative,
            detect_positive,
        }
    }
}

pub fn is_inf_config(curr: &Node) -> IsInfConfig {
    let mut detect_negative = true;
    let mut detect_positive = true;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "detect_negative" => detect_negative = value.clone().into_i64() != 0,
            "detect_positive" => detect_positive = value.clone().into_i64() != 0,
            _ => panic!("Unexpected attribute for IsInf: {key}"),
        }
    }

    IsInfConfig::new(detect_negative, detect_positive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(detect_negative: Option<i64>, detect_positive: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::IsInf, "test_is_inf")
            .input_tensor_f32("data", 4, None)
            .output_tensor_bool("output", 4, None);

        // Add attributes
        if let Some(v) = detect_negative {
            builder = builder.attr_int("detect_negative", v);
        }
        if let Some(v) = detect_positive {
            builder = builder.attr_int("detect_positive", v);
        }

        builder.build()
    }

    #[test]
    fn test_is_inf_config_default() {
        let node = create_test_node(None, None);
        let config = is_inf_config(&node);

        // Both should default to true if not specified according to the spec
        assert!(config.detect_negative);
        assert!(config.detect_positive);
    }

    #[test]
    fn test_is_inf_only_neg() {
        let node = create_test_node(Some(1), Some(0));
        let config = is_inf_config(&node);

        assert!(config.detect_negative);
        assert!(!config.detect_positive);
    }

    #[test]
    fn test_is_inf_only_pos() {
        let node = create_test_node(Some(0), Some(1));
        let config = is_inf_config(&node);

        assert!(!config.detect_negative);
        assert!(config.detect_positive);
    }

    #[test]
    fn test_is_inf_detect_none() {
        let node = create_test_node(Some(0), Some(0));
        let config = is_inf_config(&node);

        assert!(!config.detect_negative);
        assert!(!config.detect_positive);
    }
}
