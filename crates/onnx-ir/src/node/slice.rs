use crate::Argument;
use crate::ir::{ArgType, Data, Node, TensorData};

/// Configuration for the Slice operation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum SliceConfig {
    /// Static slice with constant ranges known at compile time.
    Static(Vec<Option<(i64, i64)>>),
    /// Runtime slice with arguments for starts and ends determined during execution.
    Runtime {
        starts: Argument,
        ends: Argument,
        axes: Option<Argument>,
        steps: Option<Argument>,
    },
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
            None => None, // Input exists but has no static value (dynamic)
        }
    }

    // Check if we have runtime inputs (inputs without static values)
    let has_runtime_starts = node.inputs.len() > 1 && node.inputs[1].value.is_none();
    let has_runtime_ends = node.inputs.len() > 2 && node.inputs[2].value.is_none();

    // If either starts or ends are runtime, use runtime configuration
    if has_runtime_starts || has_runtime_ends {
        if node.inputs.len() < 3 {
            panic!("Slice: runtime slicing requires at least starts and ends inputs");
        }

        return SliceConfig::Runtime {
            starts: node.inputs[1].clone(),
            ends: node.inputs[2].clone(),
            axes: node.inputs.get(3).cloned(),
            steps: node.inputs.get(4).cloned(),
        };
    }

    // Handle static case (existing behavior)
    let mut starts = get_static_input_values(node, 1).unwrap_or_default();
    let mut ends = get_static_input_values(node, 2).unwrap_or_default();
    let mut axes = get_static_input_values(node, 3).unwrap_or_default();
    let mut steps = get_static_input_values(node, 4).unwrap_or_default();

    // Fall back to attributes if inputs are not provided
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "starts" => starts = value.clone().into_i64s(),
            "ends" => ends = value.clone().into_i64s(),
            "axes" => axes = value.clone().into_i64s(),
            "steps" => steps = value.clone().into_i64s(),
            _ => {}
        }
    }

    if !steps.is_empty() && steps.iter().any(|&x| x != 1) {
        panic!("Slice: steps other than 1 are not supported");
    }

    // Extract the rank of the input tensor
    let input_rank = match node.inputs.first().unwrap().clone().ty {
        crate::ir::ArgType::Tensor(tensor) => tensor.rank,
        crate::ir::ArgType::Shape(_) => 1,
        _ => panic!("Only tensor input is valid"),
    };

    // Default to all axes if not specified
    if axes.is_empty() {
        axes = (0..starts.len() as i64).collect();
    }

    // Validate input dimensions
    if starts.len() != ends.len() || starts.len() != axes.len() {
        panic!("Slice: starts, ends, and axes must have the same length");
    }

    // Convert negative axes indices to positive (counting from the end)
    for axis in &mut axes {
        if *axis < 0 {
            *axis += input_rank as i64;
        }
    }

    // Create ranges vector with None for dimensions not being sliced
    let mut ranges: Vec<Option<(i64, i64)>> = vec![None; input_rank];
    for i in 0..axes.len() {
        let axis = axes[i] as usize;
        ranges[axis] = Some((starts[i], ends[i]));
    }

    SliceConfig::Static(ranges)
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
        ArgType::Shape(_) => {
            // Slicing a Shape extracts a sub-part, resulting in a rank-1 Tensor.
            log::debug!("Slice input for {} is Shape", node.name);
            let config = slice_config(node);

            match config {
                SliceConfig::Static(ranges) => {
                    assert_eq!(
                        ranges.len(),
                        1,
                        "Slice on Shape input requires exactly one dimension slice config for node {}",
                        node.name
                    );

                    let (start, end) = ranges[0].unwrap_or_else(|| {
                        panic!(
                            "Slice config for Shape input must contain start and end indices for node {}",
                            node.name
                        )
                    });

                    let output_len = end as usize - start as usize;
                    node.outputs[0].ty = ArgType::Shape(output_len);
                }
                SliceConfig::Runtime { .. } => {
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

    fn create_test_node(
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
        use_attrs: bool,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Slice, "test_slice")
            .input_tensor_f32("data", 3, None)
            .output_default("output");

        if !use_attrs {
            // Add inputs as tensors
            builder = builder.input_tensor_i64_data("starts", starts.clone(), vec![starts.len()]);
            builder = builder.input_tensor_i64_data("ends", ends.clone(), vec![ends.len()]);

            if let Some(axes_vec) = axes.clone() {
                builder =
                    builder.input_tensor_i64_data("axes", axes_vec.clone(), vec![axes_vec.len()]);
            }
        } else {
            // Add attributes
            builder = builder.attr_ints("starts", starts);
            builder = builder.attr_ints("ends", ends);

            if let Some(axes_vec) = axes {
                builder = builder.attr_ints("axes", axes_vec);
            }
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

    #[test]
    fn test_slice_config_basic() {
        // Create a node with inputs for basic slicing
        let node = create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2]), false);

        let result = slice_config(&node);

        match result {
            SliceConfig::Static(ranges) => {
                assert_eq!(ranges.len(), 3);
                assert_eq!(ranges[0], Some((1, 3)));
                assert_eq!(ranges[1], None);
                assert_eq!(ranges[2], Some((0, 2)));
            }
            SliceConfig::Runtime { .. } => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_with_attrs() {
        // Create a node with attributes instead of inputs
        let node = create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2]), true);

        let result = slice_config(&node);

        match result {
            SliceConfig::Static(ranges) => {
                assert_eq!(ranges.len(), 3);
                assert_eq!(ranges[0], Some((1, 3)));
                assert_eq!(ranges[1], None);
                assert_eq!(ranges[2], Some((0, 2)));
            }
            SliceConfig::Runtime { .. } => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_negative_axes() {
        // Test with negative axes values
        let node = create_test_node(vec![1], vec![3], Some(vec![-3]), false);

        let result = slice_config(&node);

        match result {
            SliceConfig::Static(ranges) => {
                assert_eq!(ranges.len(), 3);
                assert_eq!(ranges[0], Some((1, 3))); // -3 -> 0 (first dimension)
                assert_eq!(ranges[1], None);
                assert_eq!(ranges[2], None);
            }
            SliceConfig::Runtime { .. } => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_default_axes() {
        // Test the default axes behavior (when axes input is not provided)
        let node = create_test_node(vec![1, 2], vec![3, 4], None, false);

        let result = slice_config(&node);

        match result {
            SliceConfig::Static(ranges) => {
                assert_eq!(ranges.len(), 3);
                assert_eq!(ranges[0], Some((1, 3)));
                assert_eq!(ranges[1], Some((2, 4)));
                assert_eq!(ranges[2], None);
            }
            SliceConfig::Runtime { .. } => panic!("Expected static config"),
        }
    }

    #[test]
    fn test_slice_config_runtime() {
        // Test with runtime inputs (no static values)
        let node = create_runtime_slice_node();

        let result = slice_config(&node);

        match result {
            SliceConfig::Runtime {
                starts,
                ends,
                axes,
                steps,
            } => {
                assert_eq!(starts.name, "starts");
                assert_eq!(ends.name, "ends");
                assert!(axes.is_some());
                assert_eq!(axes.unwrap().name, "axes");
                assert!(steps.is_some());
                assert_eq!(steps.unwrap().name, "steps");
            }
            SliceConfig::Static(_) => panic!("Expected runtime config"),
        }
    }

    #[test]
    fn test_slice_update_output_rank_tensor_input() {
        // Test when input is a Tensor - output should preserve the same type
        let mut node = create_test_node(vec![1, 2], vec![3, 4], None, false);

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
}
