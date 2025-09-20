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

/// Normalize negative axes to positive indices based on tensor rank.
fn normalize_axes(axes: &mut [i64], rank: usize, node_name: &str) {
    for axis in axes.iter_mut() {
        if *axis < 0 {
            let normalized = rank as i64 + *axis;
            log::debug!(
                "Slice node {}: normalizing negative axis {} to {} (rank={})",
                node_name,
                *axis,
                normalized,
                rank
            );
            *axis = normalized;
        }
    }
}

/// Creates a configuration for tensor slicing based on the ONNX Slice operator.
/// Returns either static ranges or runtime arguments for slicing.
///
/// Note: we leave the negative indices as is, but we need to handle them properly when slicing
/// during the actual slicing operation using the runtime shape information.
pub fn slice_config(node: &Node) -> SliceConfig {
    /// Creates a SliceInput from either a static value or runtime argument.
    fn get_slice_input(node: &Node, index: usize) -> Option<SliceInput> {
        let input = node.inputs.get(index)?;

        match &input.value {
            None => Some(SliceInput::Runtime(input.clone())),
            Some(TensorData {
                data: Data::Int64s(values),
                ..
            }) => Some(SliceInput::Static(values.clone())),
            Some(v) => panic!(
                "Slice input at index {} must be int64 but got {:?}",
                index, v
            ),
        }
    }

    let starts =
        get_slice_input(node, 1).unwrap_or_else(|| panic!("Slice: starts parameter is required"));

    let ends =
        get_slice_input(node, 2).unwrap_or_else(|| panic!("Slice: ends parameter is required"));

    let mut axes = get_slice_input(node, 3);
    let steps = get_slice_input(node, 4);

    // Validate steps if present - zeros are not allowed
    if let Some(SliceInput::Static(ref step_values)) = steps
        && step_values.contains(&0)
    {
        panic!("Slice: step values cannot be zero");
    }

    // Normalize negative axes if we have static axes and know the input rank
    if let Some(SliceInput::Static(ref mut axes_values)) = axes
        && let ArgType::Tensor(ref tensor_type) = node.inputs[0].ty
    {
        normalize_axes(axes_values, tensor_type.rank, &node.name);
    }

    SliceConfig {
        starts,
        ends,
        axes,
        steps,
    }
}

/// Calculate output length for slicing a Shape.
/// Handles negative indices, special cases, and steps.
fn calculate_shape_slice_output_len(
    start: i64,
    end: i64,
    step: i64,
    shape_rank: usize,
    node_name: &str,
) -> usize {
    let shape_len = shape_rank as i64;

    // Normalize negative indices
    let norm_start = if start < 0 {
        (shape_len + start).max(0)
    } else {
        start.min(shape_len)
    };

    // Handle special end values
    let norm_end = if end == i64::MAX || end >= shape_len {
        shape_len
    } else if end < 0 {
        (shape_len + end).max(0)
    } else {
        end.min(shape_len)
    };

    // Calculate output length considering step
    let range_len = (norm_end - norm_start).max(0);
    let output_len = if step.abs() == 1 {
        range_len as usize
    } else {
        ((range_len + step.abs() - 1) / step.abs()) as usize
    };

    log::debug!(
        "Shape slice for node {}: [{}, {}, step={}] -> [{}, {}] on rank {} = output length {}",
        node_name,
        start,
        end,
        step,
        norm_start,
        norm_end,
        shape_rank,
        output_len
    );

    // Special case logging for common patterns
    if start == -1 && (end == i64::MAX || end >= shape_len) && step == 1 {
        log::debug!(
            "Slice pattern [-1:] detected for node {} - getting last element",
            node_name
        );
    }

    output_len
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

            // Only static slicing is supported for Shape inputs
            let (starts, ends, steps) = match (&config.starts, &config.ends, &config.steps) {
                (SliceInput::Static(s), SliceInput::Static(e), steps_opt) => {
                    let step_values = match steps_opt {
                        Some(SliceInput::Static(st)) => st.clone(),
                        _ => vec![1], // Default step is 1
                    };
                    (s, e, step_values)
                }
                _ => panic!(
                    "Runtime slice on Shape input is not supported for node {}",
                    node.name
                ),
            };

            // Require exactly one dimension for Shape slicing
            if starts.len() != 1 || ends.len() != 1 {
                panic!(
                    "Slice on Shape input requires exactly one dimension slice config for node {}",
                    node.name
                );
            }

            let step = if steps.is_empty() { 1 } else { steps[0] };
            let output_len =
                calculate_shape_slice_output_len(starts[0], ends[0], step, *shape_rank, &node.name);
            node.outputs[0].ty = ArgType::Shape(output_len);
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
                // Steps should be None when not provided
                assert!(result.steps.is_none());
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
                // Check axes (should be normalized from -3 to 0 for rank 3 tensor)
                if let Some(SliceInput::Static(axes)) = &result.axes {
                    assert_eq!(axes, &vec![0]); // -3 + 3 = 0
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

    #[test]
    fn test_slice_config_with_steps() {
        // Create a node with steps input
        let builder = NodeBuilder::new(NodeType::Slice, "test_slice_with_steps")
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("starts", vec![0, 0], vec![2])
            .input_tensor_i64_data("ends", vec![10, 10], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![2, 3], vec![2])
            .output_default("output");

        let node = builder.build();
        let result = slice_config(&node);

        // Check that we have static starts, ends, and steps
        match (&result.starts, &result.ends, &result.steps) {
            (
                SliceInput::Static(starts),
                SliceInput::Static(ends),
                Some(SliceInput::Static(steps)),
            ) => {
                assert_eq!(starts, &vec![0, 0]);
                assert_eq!(ends, &vec![10, 10]);
                assert_eq!(steps, &vec![2, 3]);
            }
            _ => panic!("Expected static config with steps"),
        }
    }

    #[test]
    #[should_panic(expected = "step values cannot be zero")]
    fn test_slice_config_zero_step() {
        // Create a node with zero step value (should panic)
        let builder = NodeBuilder::new(NodeType::Slice, "test_zero_step")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0], vec![1])
            .input_tensor_i64_data("ends", vec![10], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![0], vec![1])
            .output_default("output");

        let node = builder.build();
        slice_config(&node); // Should panic
    }

    #[test]
    fn test_slice_config_negative_steps() {
        // Create a node with negative step values
        let builder = NodeBuilder::new(NodeType::Slice, "test_negative_steps")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0, 2], vec![2])
            .input_tensor_i64_data("ends", vec![10, 8], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![-1, -2], vec![2])
            .output_default("output");

        let node = builder.build();
        let result = slice_config(&node);

        // Check that negative steps are preserved
        match &result.steps {
            Some(SliceInput::Static(steps)) => {
                assert_eq!(steps, &vec![-1, -2]);
            }
            _ => panic!("Expected static steps with negative values"),
        }
    }
}
