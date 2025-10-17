use crate::ir::{ArgType, ElementType, Node, NodeConfig, RuntimeInputRef, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Represents either a static value or a runtime argument for TopK k parameter.
#[derive(Debug, Clone)]
pub enum TopKInput {
    /// Static k known at compile time.
    Static(usize),
    /// Runtime k determined during execution.
    Runtime(RuntimeInputRef),
}

/// Configuration for the TopK operation.
#[derive(Debug, Clone)]
pub struct TopKConfig {
    /// The axis along which to perform the top-k selection.
    pub axis: usize,
    /// The number of top elements to select.
    pub k: TopKInput,
}

impl NodeConfig for TopKConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct TopKProcessor;

impl NodeProcessor for TopKProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        let mut lifted = Vec::new();

        // Lift K input (input[1]) if present
        if node.inputs.len() > 1 {
            lifted.push(node.inputs[1].name.clone());
        }

        Ok(lifted)
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TopK implementation supports opset 10+ (k as input)
        crate::util::validate_opset(opset, 10)?;

        // Validate input count (1 or 2 inputs)
        crate::util::validate_min_inputs(node, 1)?;
        if node.inputs.len() > 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: node.inputs.len(),
            });
        }

        // Validate output count
        crate::util::validate_output_count(node, 2)?;

        // Validate largest and sorted attributes before config extraction
        if let Some(largest) = node.attrs.get("largest")
            && largest.clone().into_i64() != 1
        {
            return Err(ProcessError::Custom(
                "TopK: only largest elements is supported".to_string(),
            ));
        }

        if let Some(sorted) = node.attrs.get("sorted")
            && sorted.clone().into_i64() != 1
        {
            return Err(ProcessError::Custom(
                "TopK: only sorted elements is supported".to_string(),
            ));
        }

        // Extract the shape of the input data tensor
        let data_tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Infer output types
        log::debug!("TopK rank inference for node {}", node.name);

        let rank = data_tensor.rank;
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

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the shape of the input data tensor
        let data_tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        let k = match node.inputs.get(1) {
            Some(k_tensor) => match k_tensor.into_value() {
                None => {
                    // Runtime input - no static value available
                    TopKInput::Runtime(RuntimeInputRef::new(k_tensor.name.clone(), 1))
                }
                Some(tensor_data) => {
                    let k_value = tensor_data.data.into_i64s()[0];
                    TopKInput::Static(k_value as usize)
                }
            },
            _ => {
                // Fall back to attribute
                let k_value = node
                    .attrs
                    .get("k")
                    .ok_or_else(|| ProcessError::MissingAttribute("k".to_string()))?
                    .clone()
                    .into_i64();
                TopKInput::Static(k_value as usize)
            }
        };

        let mut axis = match node.attrs.get("axis") {
            Some(axis) => axis.clone().into_i64(),
            None => -1,
        };

        // If axis is negative, it is counted from the end
        if axis < 0 {
            axis += data_tensor.rank as i64;
        }

        let config = TopKConfig {
            axis: axis as usize,
            k,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType, RuntimeInputRef};
    use crate::node::test_utils::NodeBuilder;
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        attrs: Option<HashMap<String, AttributeValue>>,
        k_input_value: Option<i64>,
    ) -> NodeBuilder {
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

        builder
    }

    #[test]
    fn test_topk_basic() {
        let mut node = create_test_node(3, None, None).build();
        // Add K attribute since we didn't provide K input
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));

        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
    fn test_topk_invalid_input() {
        let mut node = create_test_node(3, None, None).build();
        node.attrs.insert("k".to_string(), AttributeValue::Int64(5));
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    // Tests for top_k_config function

    #[test]
    fn test_top_k_config_with_k_attribute() {
        // Test when k is provided as an attribute
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 10));
    }

    #[test]
    fn test_top_k_config_with_k_input() {
        // Test when k is provided as an input
        let node = create_test_node(4, None, Some(5)).build_with_graph_data(16);

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        // Default axis should be -1 which gets converted to rank-1
        assert_eq!(config.axis, 3);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_explicit_axis() {
        // Test with explicitly specified axis
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("axis".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        assert_eq!(config.axis, 1);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 3));
    }

    #[test]
    fn test_top_k_config_with_negative_axis() {
        // Test with negative axis (counts from the end)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(5));
        attrs.insert("axis".to_string(), AttributeValue::Int64(-2)); // Second-to-last axis
        let node = create_test_node(4, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        // For rank 4, axis -2 should be 2
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_largest_attribute() {
        // Test with largest attribute set to 1 (default supported behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(7));
        attrs.insert("largest".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        assert_eq!(config.axis, 1);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 7));
    }

    #[test]
    fn test_top_k_config_with_sorted_attribute() {
        // Test with sorted attribute set to 1 (default supported behavior)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(2));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(1));
        let node = create_test_node(3, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 2));
    }

    #[test]
    fn test_top_k_config_with_largest_false() {
        // Test with largest attribute set to 0 (unsupported)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("largest".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_top_k_config_with_sorted_false() {
        // Test with sorted attribute set to 0 (unsupported)
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(3));
        attrs.insert("sorted".to_string(), AttributeValue::Int64(0));
        let node = create_test_node(2, Some(attrs), None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_top_k_config_with_invalid_input_type() {
        // Test with invalid input type
        let mut node = create_test_node(2, None, None).build();
        node.attrs.insert("k".to_string(), AttributeValue::Int64(3));
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_top_k_config_without_k() {
        // Test when k is neither provided as input nor attribute
        let node = create_test_node(3, None, None).build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_top_k_config_with_both_k_input_and_attribute() {
        // Test when k is provided both as input and attribute
        // Input should take precedence
        let mut attrs = HashMap::new();
        attrs.insert("k".to_string(), AttributeValue::Int64(10));
        let node = create_test_node(3, Some(attrs), Some(5)).build_with_graph_data(16);

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        // K from input should be used (5), not from attribute (10)
        assert_eq!(config.axis, 2);
        assert!(matches!(&config.k, TopKInput::Static(k) if *k == 5));
    }

    #[test]
    fn test_top_k_config_with_runtime_k() {
        // Test when k is provided as a runtime input (no static value)
        let node = NodeBuilder::new(NodeType::TopK, "test_topk")
            .input_tensor_f32("X", 3, None)
            .input_tensor_i64("K", 0, None) // Runtime input - no static value
            .output_tensor_f32("Values", 0, None)
            .output_tensor_i64("Indices", 0, None)
            .build();

        let mut node = node;
        let processor = TopKProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<TopKConfig>();

        assert_eq!(config.axis, 2); // Default axis -1 becomes 2 for rank 3
        assert!(matches!(&config.k, TopKInput::Runtime(arg) if arg.name == "K"));
    }
}
