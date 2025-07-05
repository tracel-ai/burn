use crate::ir::{ArgType, Node, TensorType};

/// Update output rank for Split (same as input).
pub fn split_update_outputs(node: &mut Node) {
    log::debug!("Split rank inference for node {}", node.name);

    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Split: Input must be a tensor"),
    };
    log::debug!("Split input rank for {}: {}", node.name, tensor.rank);
    log::debug!(
        "Split will generate {} outputs for {}",
        node.outputs.len(),
        node.name
    );

    for (i, output_arg) in node.outputs.iter_mut().enumerate() {
        output_arg.ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
        log::debug!("Split output {} rank for {}: {}", i, node.name, tensor.rank);
    }
}

/// Configuration for the Split operation.
#[derive(Clone, Debug)]
pub struct SplitConfig {
    /// The axis along which to split the input tensor.
    pub axis: usize,
    /// The uniform size of each split when splitting evenly.
    pub split_size: Option<usize>,
    /// Custom sizes for each split when splitting unevenly.
    pub split_sizes: Option<Vec<usize>>,
}

impl SplitConfig {
    pub fn new(axis: usize, split_size: Option<usize>, split_sizes: Option<Vec<usize>>) -> Self {
        SplitConfig {
            axis,
            split_size,
            split_sizes,
        }
    }
}

/// Creates a SplitConfig from the node attributes and inputs.
pub fn split_config(node: &Node) -> SplitConfig {
    // Initialize the axis to split along (default is 0 as per ONNX specification)
    let mut axis: i64 = 0;
    // Holds the uniform split size if calculated or provided
    let mut split_size: Option<usize> = None;
    // Holds the custom split sizes if provided as input
    let mut split_sizes: Option<Vec<usize>> = None;

    // Extract the input tensor type to determine rank and shape
    let tensor = match node.inputs.first().unwrap().ty {
        ArgType::Tensor(ref tensor) => tensor,
        _ => panic!("Split: Input must be a valid tensor"),
    };

    // Optionally store the number of outputs if provided as an attribute
    let mut num_outputs: Option<usize> = None;

    // Iterate through node attributes to extract relevant values
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "num_outputs" => num_outputs = Some(value.clone().into_i64() as usize),
            _ => {}
        }
    }

    // Handle the case when num_outputs is provided to calculate uniform split size
    if let Some(num_outputs) = num_outputs {
        if num_outputs == 0 {
            panic!("Split: 'num_outputs' must be a positive value greater than zero");
        }

        let dim_size = tensor
            .static_shape
            .as_ref()
            .expect("Split: Static shape must be known to calculate split size")[axis as usize];

        // Calculate the split size considering any remainder for non-evenly divisible dimensions
        let calculated_split_size =
            dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

        if calculated_split_size == 0 {
            panic!(
                "Split: Calculated split size is zero. Please ensure 'num_outputs' is valid for the dimension size"
            );
        }

        // Assign the calculated split size
        split_size = Some(calculated_split_size);
    }

    // Adjust axis if negative to count from the end as per ONNX spec
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    // Check for custom split sizes provided as a second input
    if node.inputs.len() > 1 && node.inputs[1].value.is_some() {
        let sizes = node.inputs[1]
            .value
            .as_ref()
            .unwrap()
            .data
            .clone()
            .into_usizes();

        if !sizes.is_empty() {
            split_sizes = Some(sizes);
        }
    }

    // Ensure that only one of 'split_sizes' or 'num_outputs' is specified
    if split_sizes.is_some() && split_size.is_some() {
        panic!(
            "Split: Cannot specify both 'split' input and 'num_outputs' attribute simultaneously"
        );
    }

    // Infer split_size if neither custom split_sizes nor split_size is provided
    if split_sizes.is_none() && split_size.is_none() {
        let num_outputs = node.outputs.len();
        let dim_size = tensor
            .static_shape
            .as_ref()
            .expect("Split: Static shape must be known to infer split size")[axis as usize];

        // Calculate inferred split size based on number of outputs
        let calculated_split_size =
            dim_size / (num_outputs - (dim_size % num_outputs != 0) as usize);

        if calculated_split_size == 0 {
            panic!(
                "Split: Inferred split size is zero. Please ensure the number of outputs is valid for the dimension size"
            );
        }

        split_size = Some(calculated_split_size);
    }

    // Return the configuration for splitting operation
    SplitConfig {
        axis: axis as usize,
        split_size,
        split_sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, AttributeValue, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;
    use std::collections::HashMap;

    fn create_test_node(
        input_rank: usize,
        num_outputs: usize,
        static_shape: Option<Vec<usize>>,
        attrs: Option<HashMap<String, AttributeValue>>,
        split_sizes_input: Option<Vec<i64>>,
    ) -> Node {
        // Start with input tensor
        let mut builder = NodeBuilder::new(NodeType::Split, "test_split").input_tensor_f32(
            "input",
            input_rank,
            static_shape,
        );

        // Add split sizes input if provided
        if let Some(sizes) = split_sizes_input {
            builder = builder.input_tensor_i64_data("split", sizes.clone(), vec![sizes.len()]);
        }

        // Add output tensors
        for i in 0..num_outputs {
            builder = builder.output_tensor_f32(
                &format!("output_{i}"),
                0, // Will be updated
                None,
            );
        }

        // Add attributes if provided
        let mut node = builder.build();

        if let Some(attributes) = attrs {
            node.attrs = attributes;
        }

        node
    }

    #[test]
    fn test_split_single_output() {
        let mut node = create_test_node(3, 1, None, None, None);
        split_update_outputs(&mut node);

        assert_eq!(node.outputs.len(), 1);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_split_multiple_outputs() {
        let mut node = create_test_node(4, 3, None, None, None);
        split_update_outputs(&mut node);

        assert_eq!(node.outputs.len(), 3);
        for output in &node.outputs {
            match &output.ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(tensor.elem_type, ElementType::Float32);
                    assert_eq!(tensor.rank, 4);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "Split: Input must be a tensor")]
    fn test_split_invalid_input() {
        let mut node = create_test_node(3, 2, None, None, None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        split_update_outputs(&mut node);
    }

    // Tests for split_config function

    #[test]
    fn test_split_config_default_axis() {
        // Create a node with static shape and 2 outputs
        let static_shape = Some(vec![10, 20, 30]);
        let node = create_test_node(3, 2, static_shape, None, None);

        let config = split_config(&node);

        // Default axis should be 0, and split_size should be calculated
        assert_eq!(config.axis, 0);
        assert_eq!(config.split_size, Some(5)); // 10 / 2 = 5
        assert_eq!(config.split_sizes, None);
    }

    #[test]
    fn test_split_config_specified_axis() {
        // Create a node with static shape, 2 outputs, and a specified axis
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(1)); // Split along axis 1

        let node = create_test_node(3, 2, static_shape, Some(attrs), None);

        let config = split_config(&node);

        assert_eq!(config.axis, 1);
        assert_eq!(config.split_size, Some(10)); // 20 / 2 = 10
        assert_eq!(config.split_sizes, None);
    }

    #[test]
    fn test_split_config_negative_axis() {
        // Test with negative axis (should count from the end)
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(-1)); // Last axis (index 2)

        let node = create_test_node(3, 3, static_shape, Some(attrs), None);

        let config = split_config(&node);

        assert_eq!(config.axis, 2); // -1 should be converted to 2
        assert_eq!(config.split_size, Some(10)); // 30 / 3 = 10
        assert_eq!(config.split_sizes, None);
    }

    #[test]
    fn test_split_config_num_outputs_attr() {
        // Test with explicitly specified num_outputs attribute
        let static_shape = Some(vec![12, 24, 36]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(4));

        let node = create_test_node(3, 4, static_shape, Some(attrs), None);

        let config = split_config(&node);

        assert_eq!(config.axis, 0);
        assert_eq!(config.split_size, Some(3)); // 12 / 4 = 3
        assert_eq!(config.split_sizes, None);
    }

    #[test]
    fn test_split_config_with_split_sizes_input() {
        // Test with explicit split sizes provided as second input
        let static_shape = Some(vec![10, 20, 30]);
        let split_sizes = vec![5, 15]; // Custom split sizes along default axis

        let node = create_test_node(3, 2, static_shape, None, Some(split_sizes.clone()));

        let config = split_config(&node);

        assert_eq!(config.axis, 0);
        assert_eq!(config.split_size, None);
        assert_eq!(config.split_sizes, Some(vec![5, 15]));
    }

    #[test]
    #[should_panic(
        expected = "Split: Cannot specify both 'split' input and 'num_outputs' attribute simultaneously"
    )]
    fn test_split_config_both_splits_and_num_outputs() {
        // Test with both split sizes input and num_outputs attribute (should panic)
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(2));
        let split_sizes = vec![3, 7];

        let node = create_test_node(3, 2, static_shape, Some(attrs), Some(split_sizes));

        let _ = split_config(&node);
    }

    #[test]
    #[should_panic(expected = "Split: 'num_outputs' must be a positive value greater than zero")]
    fn test_split_config_zero_num_outputs() {
        // Test with num_outputs attribute set to 0 (should panic)
        let static_shape = Some(vec![10, 20, 30]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(0));

        let node = create_test_node(3, 0, static_shape, Some(attrs), None);

        let _ = split_config(&node);
    }

    #[test]
    #[should_panic(expected = "Split: Calculated split size is zero")]
    fn test_split_config_invalid_num_outputs() {
        // Test with num_outputs larger than the dimension size (should result in split_size = 0)
        let static_shape = Some(vec![5, 10, 15]);
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(10)); // Larger than dim 0 size

        let node = create_test_node(3, 10, static_shape, Some(attrs), None);

        let _ = split_config(&node);
    }

    #[test]
    #[should_panic(expected = "Split: Static shape must be known to calculate split size")]
    fn test_split_config_no_static_shape() {
        // Test with no static shape available
        let mut attrs = HashMap::new();
        attrs.insert("num_outputs".to_string(), AttributeValue::Int64(2));

        let node = create_test_node(3, 2, None, Some(attrs), None);

        let _ = split_config(&node);
    }

    #[test]
    #[should_panic(expected = "Split: Input must be a valid tensor")]
    fn test_split_config_invalid_input_type() {
        // Test with invalid input type
        let mut node = create_test_node(3, 2, Some(vec![10, 20, 30]), None, None);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);

        let _ = split_config(&node);
    }

    #[test]
    fn test_split_config_non_even_split() {
        // Test with non-evenly divisible dimension size
        let static_shape = Some(vec![11, 22, 33]); // 11 is not evenly divisible by 3
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(0));

        let node = create_test_node(3, 3, static_shape, Some(attrs), None);

        let config = split_config(&node);

        // 11 / (3-1) = 5, since the dimension is not evenly divisible
        assert_eq!(config.split_size, Some(5));
    }
}
