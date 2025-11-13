//! # Slice
//!
//! Extracts a slice from the input tensor along multiple axes.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Slice.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)
//! - Tind: tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 1-9**: `starts`, `ends`, and `axes` were attributes (static only).
//! - **Opset 10**: **BREAKING CHANGE** - Changed `starts`, `ends`, and `axes` from attributes to inputs
//!   for dynamic slicing support. This enables runtime determination of slice parameters.
//! - **Opset 11**: Added optional `steps` input for strided slicing.
//! - **Opset 13**: Added bfloat16 and additional type support.

use crate::ir::{ArgType, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for the Slice operation.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    pub starts: SliceInput,
    pub ends: SliceInput,
    pub axes: Option<SliceInput>,
    pub steps: Option<SliceInput>,
}

impl NodeConfig for SliceConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for slice parameters.
#[derive(Debug, Clone)]
pub enum SliceInput {
    /// Static value known at compile time.
    Static(Vec<i64>),
    /// Runtime argument determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Normalize negative axes to positive indices based on tensor rank.
fn normalize_axes(axes: &mut [i64], rank: usize, _node_name: &str) {
    for axis in axes.iter_mut() {
        if *axis < 0 {
            *axis += rank as i64;
        }
    }
}

/// Calculate output length for slicing a Shape.
/// Handles negative indices, special cases, and steps.
fn calculate_shape_slice_output_len(
    start: i64,
    end: i64,
    step: i64,
    shape_rank: usize,
    _node_name: &str,
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

    if step.abs() == 1 {
        range_len as usize
    } else {
        ((range_len + step.abs() - 1) / step.abs()) as usize
    }
}

pub struct SliceProcessor;

impl NodeProcessor for SliceProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::AtLeast(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Option<crate::processor::InputPreferences>, ProcessError> {
        use crate::processor::{ArgPreference, InputPreferences};

        let mut prefs = InputPreferences::new();
        for input in node.inputs.iter().skip(1) {
            // Prefer this constant to be Shape
            prefs = prefs.add(&input.name, ArgPreference::Shape);
        }

        Ok(Some(prefs))
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Lift starts input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift ends input (input[2]) if present
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift axes input (input[3]) if present
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        // Lift steps input (input[4]) if present
        if node.inputs.len() > 4 && node.inputs[4].is_constant() {
            node.inputs[4].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get reference to config for type inference
        let config = node.config::<SliceConfig>();

        // Infer output type based on input type
        let input_ty = node.inputs[0].ty.clone();

        match input_ty {
            ArgType::Tensor(_) => {
                // Slicing a tensor preserves its type and rank during rank inference.
                // Shape inference pass will handle the actual shape changes.
                node.outputs[0].ty = input_ty;
            }
            ArgType::Shape(shape_rank) => {
                // Slicing a Shape extracts a sub-part, resulting in a Shape.
                // Only static slicing is supported for Shape inputs
                let (starts, ends, steps) = match (&config.starts, &config.ends, &config.steps) {
                    (SliceInput::Static(s), SliceInput::Static(e), steps_opt) => {
                        let step_values = match steps_opt {
                            Some(SliceInput::Static(st)) => st.clone(),
                            _ => vec![1], // Default step is 1
                        };
                        (s, e, step_values)
                    }
                    _ => {
                        return Err(ProcessError::Custom(format!(
                            "Runtime slice on Shape input is not supported for node {}",
                            node.name
                        )));
                    }
                };

                // Require exactly one dimension for Shape slicing
                if starts.len() != 1 || ends.len() != 1 {
                    return Err(ProcessError::Custom(format!(
                        "Slice on Shape input requires exactly one dimension slice config for node {}",
                        node.name
                    )));
                }

                let step = if steps.is_empty() { 1 } else { steps[0] };
                let output_len = calculate_shape_slice_output_len(
                    starts[0], ends[0], step, shape_rank, &node.name,
                );
                node.outputs[0].ty = ArgType::Shape(output_len);
            }
            unsupported_type => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", unsupported_type),
                });
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract config - helper function to get slice inputs
        fn get_slice_input(
            node: &NodeBuilder,
            index: usize,
        ) -> Result<Option<SliceInput>, ProcessError> {
            let input = match node.inputs.get(index) {
                Some(i) => i,
                None => return Ok(None),
            };

            // Check if this is a Shape type (preferred for Slice parameters)
            if matches!(input.ty, ArgType::Shape(_)) {
                // Try to get static value if available
                match input.value() {
                    Some(tensor_data) => match tensor_data.to_i64_vec() {
                        Ok(vec) => return Ok(Some(SliceInput::Static(vec))),
                        Err(_) => {
                            return Err(ProcessError::Custom(format!(
                                "Slice Shape input at index {} must be int32 or int64",
                                index
                            )));
                        }
                    },
                    None => {
                        // Shape type without value means it's a runtime Shape (e.g., from Shape node)
                        // Runtime input - store reference instead of cloning the argument
                        return Ok(Some(SliceInput::Runtime(RuntimeInputRef::new(
                            input.name.clone(),
                            index,
                        ))));
                    }
                }
            }

            // Otherwise, handle as Tensor (backward compatibility)
            match input.value() {
                None => {
                    // Runtime input - store reference instead of cloning the argument
                    Ok(Some(SliceInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        index,
                    ))))
                }
                Some(tensor_data) => match tensor_data.to_i64_vec() {
                    Ok(vec) => Ok(Some(SliceInput::Static(vec))),
                    Err(_) => Err(ProcessError::Custom(format!(
                        "Slice input at index {} must be int32 or int64",
                        index
                    ))),
                },
            }
        }

        let starts = get_slice_input(node, 1)?
            .ok_or_else(|| ProcessError::MissingInput("starts".to_string()))?;

        let ends = get_slice_input(node, 2)?
            .ok_or_else(|| ProcessError::MissingInput("ends".to_string()))?;

        let mut axes = get_slice_input(node, 3)?;
        let steps = get_slice_input(node, 4)?;

        // Validate steps if present - zeros are not allowed
        if let Some(SliceInput::Static(ref step_values)) = steps
            && step_values.contains(&0)
        {
            return Err(ProcessError::Custom(
                "Slice: step values cannot be zero".to_string(),
            ));
        }

        // TODO: Missing validation that starts, ends, axes, and steps have same length.
        // ONNX spec requires: len(starts) == len(ends) == len(axes) == len(steps).
        // Implementation doesn't validate this constraint, which could cause indexing errors.

        // TODO: Missing validation that axes values are unique and in valid range [-rank, rank-1].
        // Duplicate axes or out-of-range axes should be rejected but aren't validated.

        // Normalize negative axes if we have static axes and know the input rank
        if let Some(SliceInput::Static(ref mut axes_values)) = axes
            && let ArgType::Tensor(ref tensor_type) = node.inputs[0].ty
        {
            normalize_axes(axes_values, tensor_type.rank, &node.name);
        }

        let config = SliceConfig {
            starts,
            ends,
            axes,
            steps,
        };
        Ok(Some(Box::new(config)))
    }

    fn build_node(&self, builder: NodeBuilder) -> Node {
        let config = builder
            .config
            .expect("Config should be set by extract_config")
            .as_any()
            .downcast_ref::<SliceConfig>()
            .expect("Wrong config type")
            .clone();

        Node::Slice {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    use super::*;

    fn create_test_node(
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Slice, "test_slice")
            .input_tensor_f32("data", 3, None)
            .output_default("output");

        // Add inputs as tensors
        builder = builder.input_tensor_i64_data("starts", starts.clone(), vec![starts.len()]);
        builder = builder.input_tensor_i64_data("ends", ends.clone(), vec![ends.len()]);

        if let Some(axes_vec) = axes.clone() {
            builder = builder.input_tensor_i64_data("axes", axes_vec.clone(), vec![axes_vec.len()]);
        }

        builder
    }

    fn create_shape_input_node(start: i64, end: i64) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_slice_shape")
            .input_shape("data", 5)
            .input_tensor_i64_data("starts", vec![start], vec![1])
            .input_tensor_i64_data("ends", vec![end], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .output_default("output")
    }

    fn create_runtime_slice_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_runtime_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // No static value - runtime input
            .input_tensor_i64("ends", 0, None) // No static value - runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    fn create_mixed_slice_node_runtime_start() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("starts", 0, None) // Runtime input
            .input_tensor_i64_data("ends", vec![3], vec![1]) // Static input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    fn create_mixed_slice_node_runtime_end() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Slice, "test_mixed_slice")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![1], vec![1]) // Static input
            .input_tensor_i64("ends", 0, None) // Runtime input
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![1], vec![1])
            .output_default("output")
    }

    #[test]
    fn test_slice_config_basic() {
        // Create a node with inputs for basic slicing
        let node =
            create_test_node(vec![1, 0], vec![3, 2], Some(vec![0, 2])).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let node = create_test_node(vec![1], vec![3], Some(vec![-3])).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let node = create_test_node(vec![1, 2], vec![3, 4], None).build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let node = create_runtime_slice_node().build();

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let mut node = create_test_node(vec![1, 2], vec![3, 4], None).build_with_graph_data(16);

        // Before calling, input is Tensor and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Tensor(_)));
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(_)));

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // After calling, output should be the same type as input
        assert!(
            matches!(&node.outputs[0].ty, ArgType::Tensor(tensor_type) if tensor_type.dtype == DType::F32 && tensor_type.rank == 3)
        );
    }

    #[test]
    fn test_slice_update_output_rank_shape_input() {
        // Test when input is a Shape - output should be a rank-1 Int64 Tensor
        let mut node = create_shape_input_node(1, 3).build_with_graph_data(16);

        // Before calling, input is Shape and output is default
        assert!(matches!(node.inputs[0].ty, ArgType::Shape(5)));
        // Default output type is Tensor with rank 0
        assert!(matches!(node.outputs[0].ty, ArgType::Tensor(ref t) if t.rank == 0));

        let processor = SliceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // After calling, output should be ArgType::Shape with the calculated length
        // start = 1, end = 3 => output_len = 3 - 1 = 2
        assert!(matches!(&node.outputs[0].ty, ArgType::Shape(2)));
    }

    #[test]
    fn test_slice_config_mixed_runtime_start() {
        // Test with runtime start but static end
        let node = create_mixed_slice_node_runtime_start().build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let node = create_mixed_slice_node_runtime_end().build_with_graph_data(16);

        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_slice_with_steps")
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("starts", vec![0, 0], vec![2])
            .input_tensor_i64_data("ends", vec![10, 10], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![2, 3], vec![2])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

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
    fn test_slice_config_zero_step() {
        // Create a node with zero step value (should return error)
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_zero_step")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0], vec![1])
            .input_tensor_i64_data("ends", vec![10], vec![1])
            .input_tensor_i64_data("axes", vec![0], vec![1])
            .input_tensor_i64_data("steps", vec![0], vec![1])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let node = node;

        let processor = SliceProcessor;

        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_slice_config_negative_steps() {
        // Create a node with negative step values
        let builder = TestNodeBuilder::new(NodeType::Slice, "test_negative_steps")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("starts", vec![0, 2], vec![2])
            .input_tensor_i64_data("ends", vec![10, 8], vec![2])
            .input_tensor_i64_data("axes", vec![0, 1], vec![2])
            .input_tensor_i64_data("steps", vec![-1, -2], vec![2])
            .output_default("output");

        let node = builder.build_with_graph_data(16);
        let mut node = node;

        let processor = SliceProcessor;

        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let result = node.config::<SliceConfig>();

        // Check that negative steps are preserved
        match &result.steps {
            Some(SliceInput::Static(steps)) => {
                assert_eq!(steps, &vec![-1, -2]);
            }
            _ => panic!("Expected static steps with negative values"),
        }
    }

    // TODO: Missing test for mismatched lengths of starts/ends/axes/steps.
    // ONNX spec requires same length but this isn't tested or validated.

    // TODO: Missing test for duplicate axes - e.g., axes=[0, 0] should be invalid.
    // Spec doesn't allow duplicate axes but implementation doesn't validate this.

    // TODO: Missing test for out-of-bounds axes after normalization.
    // E.g., for rank-3 tensor, axes=[5] should be invalid even after negative index handling.

    // TODO: Missing test for slicing with negative steps and various start/end combinations.
    // Negative steps reverse the slice direction, need comprehensive tests for this.

    // TODO: Missing test for empty slice results - when start >= end with positive step.
    // Should result in zero-size dimension along that axis.

    // TODO: Missing test for INT32_MAX / INT64_MAX special values in ends.
    // ONNX spec mentions these special values mean "slice to the end" but not tested.
}
