//! # Clip
//!
//! Clips (limits) the values in the input tensor to a specified min/max range. Values below the minimum
//! are set to the minimum value, and values above the maximum are set to the maximum value.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Clip.html>
//!
//! ## Attributes
//!
//! - `min` (float, optional, Opset 6-10 only): Minimum value. Elements below this are clipped to min.
//! - `max` (float, optional, Opset 6-10 only): Maximum value. Elements above this are clipped to max.
//!
//! Note: At least one of `min` or `max` must be specified. If both are provided and min > max,
//! all values are set to max.
//!
//! ## Inputs
//!
//! - `input` (T): Input tensor whose elements will be clipped.
//! - `min` (T, optional, Opset 11+): Minimum value as a scalar tensor. Default: numeric_limits::lowest()
//! - `max` (T, optional, Opset 11+): Maximum value as a scalar tensor. Default: numeric_limits::max()
//!
//! ## Outputs
//!
//! - `output` (T): Output tensor with the same shape and type as input, containing clipped values.
//!
//! ## Type Constraints
//!
//! - T: tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(int16), tensor(int32),
//!   tensor(int64), tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(bfloat16)
//!
//! ## Opset Versions
//!
//! - **Opset 6**: Initial version with `min` and `max` as attributes (float values only).
//! - **Opset 11**: Changed `min` and `max` from attributes to optional inputs, allowing runtime values.
//! - **Opset 12**: Extended type support to include integer types (int8, int16, int32, int64, uint8, uint16, uint32, uint64).
//! - **Opset 13**: Added bfloat16 support. Defined behavior when min > max: all values set to max.

use crate::ir::{Data, Node, NodeConfig, RuntimeInputRef};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError, same_as_input};
use std::any::Any;

/// Represents either a static value or a runtime argument for clip parameters.
#[derive(Debug, Clone)]
pub enum ClipInput {
    /// Static value known at compile time.
    Static(f64),
    /// Runtime argument determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Configuration for Clip operation
#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub min: Option<ClipInput>,
    pub max: Option<ClipInput>,
}

impl NodeConfig for ClipConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ClipProcessor;

impl NodeProcessor for ClipProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift min (input[1]) and max (input[2]) if present and they have constant values
        // For Opset 6-10: min/max are attributes, not inputs (no lifting needed)
        // For Opset 11+: min/max are optional inputs that might be constants or runtime values
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 6)?;

        // Validate input count (at least 1 input, up to 3 for opset 11+)
        crate::processor::validate_min_inputs(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Infer output type
        same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        fn get_clip_input(node: &Node, index: usize, _param_name: &str) -> Option<ClipInput> {
            let input = node.inputs.get(index)?;

            // In ONNX, optional inputs are represented by empty strings
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return None;
            }

            match input.value() {
                None => {
                    // Runtime input - store reference instead of cloning the argument
                    Some(ClipInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        index,
                    )))
                }
                Some(tensor_data) => {
                    // Static input - extract the scalar value
                    let scalar = tensor_data.data.into_scalar();
                    let value = match scalar {
                        Data::Float16(v) => f32::from(v) as f64,
                        Data::Float32(v) => v as f64,
                        Data::Float64(v) => v,
                        Data::Int32(v) => v as f64,
                        Data::Int64(v) => v as f64,
                        _ => {
                            // TODO: According to spec (opset 12+), Clip supports additional integer types:
                            // int8, int16, uint8, uint16, uint32, uint64
                            // and bfloat16 (opset 13+)
                            return None; // Unsupported type
                        }
                    };
                    Some(ClipInput::Static(value))
                }
            }
        }

        let mut min_result: Option<ClipInput> = None;
        let mut max_result: Option<ClipInput> = None;

        // For Clip Opset 6+, the min and max values are attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "min" => {
                    let min = value.clone().into_f32() as f64;
                    min_result = Some(ClipInput::Static(min));
                }
                "max" => {
                    let max = value.clone().into_f32() as f64;
                    max_result = Some(ClipInput::Static(max));
                }
                _ => {}
            }
        }

        // For Clip Opset 11+, the min and max values are inputs
        // Check if inputs are available and attributes weren't set
        if min_result.is_none() {
            min_result = get_clip_input(node, 1, "min");
        }

        if max_result.is_none() {
            max_result = get_clip_input(node, 2, "max");
        }

        // Validate that at least one of min or max is specified
        if min_result.is_none() && max_result.is_none() {
            return Err(ProcessError::Custom(
                "Clip operation requires at least one of min or max to be specified".to_string(),
            ));
        }

        let config = ClipConfig {
            min: min_result,
            max: max_result,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node_with_attributes(min: Option<f32>, max: Option<f32>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None);

        if let Some(min_val) = min {
            builder = builder.attr_float("min", min_val);
        }

        if let Some(max_val) = max {
            builder = builder.attr_float("max", max_val);
        }

        builder.build()
    }

    fn create_test_node_with_inputs(min: Option<f32>, max: Option<f32>) -> NodeBuilder {
        // In ONNX Clip Opset 11+, inputs are positional:
        // Input 0: input
        // Input 1: min (optional)
        // Input 2: max (optional)
        // We need to maintain the correct positions even if values are None
        let builder = NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_scalar_tensor_f32("min", min)
            .input_scalar_tensor_f32("max", max)
            .output_tensor_f32("Y", 4, None);

        builder
    }

    #[test]
    fn test_clip_config_with_attributes() {
        let node = create_test_node_with_attributes(Some(-1.0), Some(1.0));
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        assert!(matches!(config.min, Some(ClipInput::Static(v)) if (v - (-1.0)).abs() < 1e-6));
        assert!(matches!(config.max, Some(ClipInput::Static(v)) if (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_clip_config_with_attributes_min_only() {
        let node = create_test_node_with_attributes(Some(-1.0), None);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        assert!(matches!(config.min, Some(ClipInput::Static(v)) if (v - (-1.0)).abs() < 1e-6));
        assert!(config.max.is_none());
    }

    #[test]
    fn test_clip_config_with_attributes_max_only() {
        let node = create_test_node_with_attributes(None, Some(1.0));
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        assert!(config.min.is_none());
        assert!(matches!(config.max, Some(ClipInput::Static(v)) if (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_clip_config_with_inputs() {
        let node = create_test_node_with_inputs(Some(-1.0), Some(1.0)).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        assert!(matches!(config.min, Some(ClipInput::Static(v)) if (v - (-1.0)).abs() < 1e-6));
        assert!(matches!(config.max, Some(ClipInput::Static(v)) if (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_clip_config_with_inputs_min_only() {
        // Note: When None is passed, input_scalar_tensor_f32 creates a runtime input
        // So this test actually has static min and runtime max
        let node = create_test_node_with_inputs(Some(-1.0), None).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        assert!(matches!(config.min, Some(ClipInput::Static(v)) if (v - (-1.0)).abs() < 1e-6));
        // max is a runtime input (no static value provided)
        assert!(matches!(config.max, Some(ClipInput::Runtime(_))));
    }

    #[test]
    fn test_clip_config_with_inputs_max_only() {
        // Note: When None is passed, input_scalar_tensor_f32 creates a runtime input
        // So this test actually has runtime min and static max
        let node = create_test_node_with_inputs(None, Some(1.0)).build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();
        // min is a runtime input (no static value provided)
        assert!(matches!(config.min, Some(ClipInput::Runtime(_))));
        assert!(matches!(config.max, Some(ClipInput::Static(v)) if (v - 1.0).abs() < 1e-6));
    }

    fn create_test_node_with_runtime_inputs() -> NodeBuilder {
        NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_tensor_f32("min", 0, None) // Runtime input - no static value
            .input_tensor_f32("max", 0, None) // Runtime input - no static value
            .output_tensor_f32("Y", 4, None)
    }

    #[test]
    fn test_clip_config_with_runtime_inputs() {
        let node = create_test_node_with_runtime_inputs().build();
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();

        // Check that we have runtime inputs
        assert!(matches!(config.min, Some(ClipInput::Runtime(ref arg)) if arg.name == "min"));
        assert!(matches!(config.max, Some(ClipInput::Runtime(ref arg)) if arg.name == "max"));
    }

    #[test]
    fn test_clip_config_mixed_static_runtime() {
        // Static min, runtime max
        let builder = NodeBuilder::new(NodeType::Clip, "test_clip")
            .input_tensor_f32("X", 4, None)
            .input_scalar_tensor_f32("min", Some(-1.0)) // Static
            .input_tensor_f32("max", 0, None) // Runtime
            .output_tensor_f32("Y", 4, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = ClipProcessor;

        // Extract config first, then infer types
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ClipConfig>();

        assert!(matches!(config.min, Some(ClipInput::Static(v)) if (v - (-1.0)).abs() < 1e-6));
        assert!(matches!(config.max, Some(ClipInput::Runtime(ref arg)) if arg.name == "max"));
    }

    #[test]
    fn test_clip_config_no_min_max() {
        let node = create_test_node_with_attributes(None, None);
        let node = node;
        let processor = ClipProcessor;

        // Extract config first - this should fail with an error
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_clip_lift_constants_with_attributes_only() {
        // Test that lift_constants doesn't try to lift when using attributes (Opset 6-10)
        let mut node = create_test_node_with_attributes(Some(-1.0), Some(1.0));
        let processor = ClipProcessor;

        // This should succeed without error since attributes are not inputs
        processor.lift_constants(&mut node, 16).unwrap();
        // Verify that no inputs were modified (node should have 1 input - the data tensor)
        assert_eq!(node.inputs.len(), 1);
    }

    #[test]
    fn test_clip_lift_constants_with_runtime_inputs() {
        // Test that lift_constants doesn't modify runtime inputs (no constant values)
        let mut node = create_test_node_with_runtime_inputs().build();
        let processor = ClipProcessor;

        // Verify inputs are not constant before lifting
        assert!(!node.inputs[1].is_constant()); // min is Dynamic, not Constant
        assert!(!node.inputs[2].is_constant()); // max is Dynamic, not Constant

        // lift_constants should succeed without modifying non-constant inputs
        processor.lift_constants(&mut node, 16).unwrap();

        // Inputs should remain unchanged (still Dynamic)
        assert!(!node.inputs[1].is_static());
        assert!(!node.inputs[2].is_static());
    }

    #[test]
    fn test_clip_lift_constants_with_static_inputs() {
        // Test that lift_constants converts constant inputs to static
        let mut node =
            create_test_node_with_inputs(Some(-1.0), Some(1.0)).build_with_graph_data(16);
        let processor = ClipProcessor;

        // Verify inputs are constant before lifting
        assert!(node.inputs[1].is_constant()); // min has constant value
        assert!(node.inputs[2].is_constant()); // max has constant value

        // Lift constants - this should convert them to Static
        processor.lift_constants(&mut node, 16).unwrap();

        // Verify inputs were converted to Static
        assert!(node.inputs[1].is_static());
        assert!(node.inputs[2].is_static());
    }
}
