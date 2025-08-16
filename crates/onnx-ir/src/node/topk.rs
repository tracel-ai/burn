use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output rank for TopK (same as input rank).
pub fn top_k_update_output(node: &mut Node) {
    log::debug!("TopK rank inference for node {}", node.name);

    let rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("TopK: invalid input type"),
    };
    log::debug!("TopK input rank for {}: {}", node.name, rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank,
        static_shape: None,
    });
    node.outputs[1].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank,
        static_shape: None,
    });

    log::debug!(
        "TopK output rank for {}: {} (both outputs)",
        node.name,
        rank
    );
}

/// Configuration for the TopK operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TopKConfig {
    /// The axis along which to perform the top-k selection.
    pub axis: usize,
    /// The number of top elements to select.
    pub k: usize,
}

impl TopKConfig {
    /// Creates a new TopKConfig.
    pub fn new(axis: usize, k: usize) -> Self {
        Self { axis, k }
    }
}

/// Creates a TopKConfig from the node attributes and inputs.
pub fn top_k_config(node: &Node) -> TopKConfig {
    // Extract the shape of the input data tensor
    let data_tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    let k = match node.inputs.get(1) {
        Some(k_tensor) => k_tensor
            .clone()
            .value
            .expect("TopK: only constant 'k' tensor is currently supported")
            .data
            .into_i64s()[0],
        _ => node
            .attrs
            .get("k")
            .expect("TopK: number of top elements 'k' is missing")
            .clone()
            .into_i64(),
    };

    let mut axis = match node.attrs.get("axis") {
        Some(axis) => axis.clone().into_i64(),
        None => -1,
    };

    // If axis is negative, it is counted from the end
    if axis < 0 {
        axis += data_tensor.rank as i64;
    }

    if let Some(largest) = node.attrs.get("largest")
        && largest.clone().into_i64() != 1
    {
        unimplemented!("TopK: only largest elements is supported")
    };

    if let Some(sorted) = node.attrs.get("sorted")
        && sorted.clone().into_i64() != 1
    {
        unimplemented!("TopK: only sorted elements is supported")
    };

    TopKConfig::new(axis as usize, k as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType};
    use crate::node::test_utils::NodeBuilder;
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        attrs: Option<HashMap<String, AttributeValue>>,
        k_input_value: Option<i64>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::TopK, "test_topk")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Values", 0, None) // Rank will be updated
            .output_tensor_i64("Indices", 0, None); // Rank will be updated

        // Add K input if provided
        if let Some(k) = k_input_value {
            builder = builder.input_tensor_i64_data("K", vec![k], vec![]);
        }

        // Add attributes if provided
        if let Some(attr_map) = attrs {
            for (key, value) in attr_map {
                match value {
                    AttributeValue::Int64(val) => builder = builder.attr_int(&key, val),
                    AttributeValue::Int64s(vals) => builder = builder.attr_ints(&key, vals),
                    AttributeValue::Float32(val) => builder = builder.attr_float(&key, val),
                    AttributeValue::Float32s(vals) => builder = builder.attr_floats(&key, vals),
                    AttributeValue::String(val) => builder = builder.attr_string(&key, &val),
                    AttributeValue::Strings(vals) => builder = builder.attr_strings(&key, vals),
                    _ => panic!("Unsupported attribute type"),
                }
            }
        }

        builder.build()
    }

    #[test]
    fn test_topk_basic() {
        let mut node = create_test_node(3, None, None);
        // Add K attribute since we didn't provide K input
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));

        top_k_update_output(&mut node);

        assert_eq!(node.outputs.len(), 2);

        // Check first output (values)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for values"),
        }

        // Check second output (indices)
        match &node.outputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for indices"),
        }
    }

    #[test]
    #[should_panic(expected = "TopK: invalid input type")]
    fn test_topk_invalid_input() {
        let mut node = create_test_node(3, None, None);
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        top_k_update_output(&mut node);
    }

    // Tests for top_k_config function

    #[test]
    fn test_top_k_config_with_k_attribute() {
        // Test when k is provided as an attribute
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), None);

        let config = top_k_config(&node);

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config, TopKConfig { axis: 2, k: 10 });
    }

    #[test]
    fn test_top_k_config_with_k_input() {
        // Test when k is provided as an input
        let node = create_test_node(4, None, Some(5));

        let config = top_k_config(&node);

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config, TopKConfig { axis: 3, k: 5 });
    }

    #[test]
    fn test_top_k_config_with_explicit_axis() {
        // Test with explicitly specified axis
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("axis".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None);

        let config = top_k_config(&node);

        assert_eq!(config, TopKConfig { axis: 1, k: 3 });
    }

    #[test]
    fn test_top_k_config_with_negative_axis() {
        // Test with negative axis (counts from the end)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(5));
        attrs.insert("axis".to_string(), AttributeValue::Int64(-2)); // Second-to-last axis
        let node = create_test_node(4, Some(attrs), None);

        let config = top_k_config(&node);

        // For rank 4, axis -2 should be 2
        assert_eq!(config, TopKConfig { axis: 2, k: 5 });
    }

    #[test]
    fn test_top_k_config_with_largest_attribute() {
        // Test with largest attribute set to 1 (default supported behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(7));
        attrs.insert("largest".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(2, Some(attrs), None);

        let config = top_k_config(&node);

        assert_eq!(config, TopKConfig { axis: 1, k: 7 });
    }

    #[test]
    fn test_top_k_config_with_sorted_attribute() {
        // Test with sorted attribute set to 1 (default supported behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(2));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None);

        let config = top_k_config(&node);

        assert_eq!(config, TopKConfig { axis: 2, k: 2 });
    }

    #[test]
    #[should_panic(expected = "only largest elements is supported")]
    fn test_top_k_config_with_largest_false() {
        // Test with largest attribute set to 0 (unsupported)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("largest".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None);

        let _ = top_k_config(&node);
    }

    #[test]
    #[should_panic(expected = "only sorted elements is supported")]
    fn test_top_k_config_with_sorted_false() {
        // Test with sorted attribute set to 0 (unsupported)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None);

        let _ = top_k_config(&node);
    }

    #[test]
    #[should_panic(expected = "Only tensor input is valid")]
    fn test_top_k_config_with_invalid_input_type() {
        // Test with invalid input type
        let mut node = create_test_node(2, None, None);
        node.attrs.insert("k".to_string(), AttributeValue::Int64(3));
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);

        let _ = top_k_config(&node);
    }

    #[test]
    #[should_panic(expected = "TopK: number of top elements 'k' is missing")]
    fn test_top_k_config_without_k() {
        // Test when k is neither provided as input nor attribute
        let node = create_test_node(3, None, None);

        let _ = top_k_config(&node);
    }

    #[test]
    fn test_top_k_config_with_both_k_input_and_attribute() {
        // Test when k is provided both as input and attribute
        // Input should take precedence
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), Some(5));

        let config = top_k_config(&node);

        // K from input should be used (5), not from attribute (10)
        assert_eq!(config, TopKConfig { axis: 2, k: 5 });
    }
}
