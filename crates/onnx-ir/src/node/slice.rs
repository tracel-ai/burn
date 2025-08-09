use crate::Argument;
use crate::ir::{ArgType, Data, Node, TensorData};

/// Configuration for the Slice operation.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    pub starts: SliceInput,
    pub ends: SliceInput,
    pub axes: Option<SliceInput>,
    pub steps: Option<SliceInput>,
}

/// Represents either a static value or a runtime argument for slice parameters.
#[derive(Debug, Clone)]
pub enum SliceInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution.
    Runtime(Argument),
}

/// Creates a configuration for tensor slicing based on the ONNX Slice operator.
/// Returns either static ranges or runtime arguments for slicing.
///
/// Note: we leave the negative indices as is, but we need to handle them properly when slicing
/// during the actual slicing operation using the runtime shape information.
pub fn slice_config(node: &Node) -> SliceConfig {
    /// Extracts int64 values from a node's input at the specified index if it has a static value.
    /// Returns None if the input doesn't exist or doesn't have a static value.
    fn get_static_input_values(node: &Node, index: usize) -> Option<Vec<i64>> {
        node.inputs.get(index)?;

        match &node.inputs[index].value {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => Some(shape.clone()),
            Some(v) => {
                panic!("Tensor data type for input at index {index} must be int64 but got {v:?}")
            }
            None => None, // Input exists but has no static value (runtime)
        }
    }

    /// Creates a SliceInput from either a static value or runtime argument.
    fn get_slice_input(node: &Node, index: usize) -> Option<SliceInput> {
        // Check if input exists
        if let Some(input) = node.inputs.get(index) {
            if input.value.is_none() {
                // Runtime input
                return Some(SliceInput::Runtime(input.clone()));
            } else {
                // Static input
                if let Some(values) = get_static_input_values(node, index) {
                    return Some(SliceInput::Static(values));
                }
            }
        }

        None
    }

    let starts =
        get_slice_input(node, 1).unwrap_or_else(|| panic!("Slice: starts parameter is required"));

    let ends =
        get_slice_input(node, 2).unwrap_or_else(|| panic!("Slice: ends parameter is required"));

    let axes = get_slice_input(node, 3);
    let steps = get_slice_input(node, 4);

    // Validate steps if present
    if let Some(SliceInput::Static(ref step_values)) = steps
        && step_values.iter().any(|&x| x != 1)
    {
        panic!("Slice: steps other than 1 are not supported");
    }

    SliceConfig {
        starts,
        ends,
        axes,
        steps,
    }
}

/// Update output type for Slice operation.
/// If the input is a Tensor, the output type remains the same.
/// If the input is a Shape, the output becomes a rank-1 Int64 Tensor representing the sliced dimension.
pub fn slice_update_output_rank(node: &mut Node) {
    log::debug!("Slice rank inference for node {}", node.name);

    match &node.inputs[0].ty {
        ArgType::Tensor(_) => {
            // Slicing a tensor preserves its type and rank during rank inference.
            // Shape inference pass will handle the actual shape changes.
            log::debug!("Slice input for {} is Tensor, preserving type", node.name);
            node.outputs[0].ty = node.inputs[0].ty.clone();
        }
        ArgType::Shape(shape_rank) => {
            // Slicing a Shape extracts a sub-part, resulting in a rank-1 Tensor.
            log::debug!("Slice input for {} is Shape", node.name);
            let config = slice_config(node);

            // Check if both starts and ends are static
            match (&config.starts, &config.ends) {
                (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                    if starts.len() != 1 || ends.len() != 1 {
                        panic!(
                            "Slice on Shape input requires exactly one dimension slice config for node {}",
                            node.name
                        );
                    }

                    let start = starts[0];
                    let end = ends[0];

                    // Special case: slice[-1:] is common for getting the last dimension
                    if start == -1 && (end == i64::MAX || end >= *shape_rank as i64) {
                        // This gets the last element, output is Shape(1)
                        node.outputs[0].ty = ArgType::Shape(1);
                        log::debug!(
                            "Slice on Shape with [-1:] pattern for node {}, output: Shape(1)",
                            node.name
                        );
                    } else if start < 0 || end < 0 {
                        // Handle negative indices - convert to positive for size calculation
                        // For shapes, we know the rank at compile time
                        let shape_len = *shape_rank as i64;
                        let pos_start = if start < 0 { shape_len + start } else { start };
                        let pos_end = if end < 0 { shape_len + end } else { end };

                        if pos_start >= 0 && pos_end >= 0 && pos_end > pos_start {
                            let output_len = (pos_end - pos_start) as usize;
                            node.outputs[0].ty = ArgType::Shape(output_len);
                            log::debug!(
                                "Slice on Shape with negative indices (start={}, end={}) for node {}, output: Shape({})",
                                start,
                                end,
                                node.name,
                                output_len
                            );
                        } else {
                            // Invalid slice bounds, keep original shape rank as fallback
                            log::warn!(
                                "Slice on Shape with negative indices (start={}, end={}) has invalid bounds for node {}. Keeping original rank.",
                                start,
                                end,
                                node.name
                            );
                            node.outputs[0].ty = ArgType::Shape(*shape_rank);
                        }
                    } else {
                        // Positive indices
                        let normalized_end = if end == i64::MAX || end >= *shape_rank as i64 {
                            *shape_rank
                        } else {
                            end as usize
                        };

                        let output_len = normalized_end.saturating_sub(start as usize);
                        node.outputs[0].ty = ArgType::Shape(output_len);
                    }
                }
                _ => {
                    // For runtime slice on Shape, we can't determine the output size at compile time
                    panic!(
                        "Runtime slice on Shape input is not supported for node {}",
                        node.name
                    );
                }
            }
        }
        // Handle unsupported input types
        unsupported_type => {
            panic!(
                "Slice: Only Tensor and Shape inputs are supported for node {}, got {:?}",
                node.name, unsupported_type
            )
        }
    }

    log::debug!(
        "Slice output type determined for {}: {:?}",
        node.name,
        node.outputs[0].ty
    );
}

#[cfg(test)]
mod tests {
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    use super::*;

    fn create_test_node(starts: Vec<i64>, ends: Vec<i64>, axes: Option<Vec<i64>>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Slice, "test_slice")
            .input_tensor_f32("data", 3, None)
            .output_default("output");

        // Add inputs as tensors
        builder = builder.input_tensor_i64_data("starts", starts.clone(), vec![starts.len()]);
        builder = builder.input_tensor_i64_data("ends", ends.clone(), vec![ends.len()]);

        if let Some(axes_vec) = axes.clone() {
            builder = builder.input_tensor_i64_data("axes", axes_vec.clone(), vec![axes_vec.len()]);
        }

        builder.build()
    }

    fn create_shape_input_node(start: i64, end: i64) -> Node {
        NodeBuilder::new(NodeType::Slice, "test_slice_shape")
            .input_shape("data", 5)
            .input_tensor_i64_data("starts", vec![start], vec![1])
            .input_tensor_i64_data("ends", vec![end], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .output_default("output")
            .build()
    }

    fn create_runtime_slice_node() -> Node {
        NodeBuilder::new(NodeType::Slice, "test_runtime_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // No static value - runtime input
            .input_tensor_i64("ends", 0, None) // No static value - runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
            .build()
    }

    fn create_mixed_slice_node_runtime_start() -> Node {
        NodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // Runtime input
            .input_tensor_i64_data("ends", vec![3], vec![1]) // Static input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
            .build()
    }

    fn create_mixed_slice_node_runtime_end() -> Node {
        NodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![1], vec![1]) // Static input
            .input_tensor_i64("ends", 0, None) // Runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
            .build()
    }

    #[test]
    fn test_slice_config_basic() {
        // Create a node with inputs for basic slicing
        let node = create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2]));

        let result = slice_config(&node);

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1, 0]);
                assert_eq!(ends, &vec![3, 2]);
                // Check axes
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0, 2]);
                }
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_negative_axes() {
        // Test with negative axes values
        let node = create_test_node(vec![1], vec![3], Some(vec![-3]));

        let result = slice_config(&node);

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1]);
                assert_eq!(ends, &vec![3]);
                // Check axes (should still be negative - conversion happens in burn-import)
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![-3]);
                }
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_default_axes() {
        // Test the default axes behavior (when axes input is not provided)
        let node = create_test_node(vec![1, 2], vec![3, 4], None);

        let result = slice_config(&node);

        // Check that we have static starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts, &vec![1, 2]);
                assert_eq!(ends, &vec![3, 4]);
                // axes should be None when not provided
                assert!(result.axes.is_none());
            }
            _ => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_runtime() {
        // Test with runtime inputs (no static values)
        let node = create_runtime_slice_node();

        let result = slice_config(&node);

        // Check that we have runtime starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Runtime(starts), SliceInput::Runtime(ends)) => {
                assert_eq!(starts.name, "starts");
                assert_eq!(ends.name, "ends");
                // Check axes and steps
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0]);
                }
                if let Some(SliceInput::Static(steps)) = &result.steps {
                    assert_eq!(steps, &vec![1]);
                }
            }
            _ => panic!("Expected runtime config"),
        }
    }

    #[test]
    fn test_slice_update_output_rank_tensor_input() {
        // Test when input is a Tensor - output should preserve the same type
        let mut node = create_test_node(vec![1, 2], vec![3, 4], None);

        // Before calling, input is Tensor and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Tensor(_)));
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(_)));

        slice_update_output_rank(&mut node);

        // After calling, output should be the same type as input
        assert!(
            matches!(&node.outputs[0].ty, ArgType::Tensor(tensor_type) if tensor_type.elem_type == ElementType::Float32 && tensor_type.rank == 3)
        );
    }

    #[test]
    fn test_slice_update_output_rank_shape_input() {
        // Test when input is a Shape - output should be a rank-1 Int64 Tensor
        let mut node = create_shape_input_node(1, 3);

        // Before calling, input is Shape and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Shape(5)));
        // Default output type is Tensor with rank 0
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(ref t) if t.rank == 0));

        slice_update_output_rank(&mut node);

        // After calling, output should be ArgType::Shape with the calculated length
        // start = 1, end = 3 => output_len = 3 - 1 = 2
        assert!(matches!(&node.outputs[0].ty, ArgType::Shape(2)));
    }

    #[test]
    fn test_slice_config_mixed_runtime_start() {
        // Test with runtime start but static end
        let node = create_mixed_slice_node_runtime_start();

        let result = slice_config(&node);

        // Check that we have mixed starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Runtime(starts), SliceInput::Static(ends)) => {
                assert_eq!(starts.name, "starts");
                assert_eq!(ends, &vec![3]);
            }
            _ => panic!("Expected mixed config with runtime start and static end"),
        }
    }

    #[test]
    fn test_slice_config_mixed_runtime_end() {
        // Test with static start but runtime end
        let node = create_mixed_slice_node_runtime_end();

        let result = slice_config(&node);

        // Check that we have mixed starts and ends
        match (&result.starts, &result.ends) {
            (SliceInput::Static(starts), SliceInput::Runtime(ends)) => {
                assert_eq!(starts, &vec![1]);
                assert_eq!(ends.name, "ends");
            }
            _ => panic!("Expected mixed config with static start and runtime end"),
        }
    }
}
