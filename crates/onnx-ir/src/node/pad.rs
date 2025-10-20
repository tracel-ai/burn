//! # Pad
//!
//! Pads input tensor with additional values at borders.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Pad.html>
//!
//! ## Attributes
//! - `mode` (string, default="constant"): Padding mode (constant/reflect/edge, only constant supported)
//!
//! ## Inputs
//! - `data` (T): Input tensor
//! - `pads` (tensor(int64)): Padding amounts \[x1_begin, x2_begin, ..., x1_end, x2_end, ...\]
//! - `constant_value` (T, optional): Constant fill value, default 0
//! - `axes` (tensor(int64), optional): Axes to apply pads (not supported)
//!
//! ## Outputs
//! - `output` (T): Padded tensor
//!
//! ## Opset Versions
//! - Opset 11+

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use crate::ir::{ArgType, AttributeValue, Data, Node, NodeConfig, RuntimeInputRef};
use std::any::Any;

/// Represents either a static value or a runtime argument for pad values.
#[derive(Debug, Clone)]
pub enum PadInput {
    /// Static pads known at compile time.
    Static(Vec<usize>),
    /// Runtime pads determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Represents either a static value or a runtime argument for constant value.
#[derive(Debug, Clone)]
pub enum ConstantValueInput {
    /// Static constant value known at compile time.
    Static(f32),
    /// Runtime constant value determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

/// Configuration for the Pad operation.
#[derive(Debug, Clone)]
pub struct PadConfig {
    /// The paddings to be applied to each dimension.
    pub pads: PadInput,
    /// The constant value to fill the padded areas with.
    pub constant_value: ConstantValueInput,
}

impl NodeConfig for PadConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct PadProcessor;

impl NodeProcessor for PadProcessor {
    // TODO mark axes inputs as Shape if inputs are constant

    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift pads input (input[1]) if present
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift constant_value input (input[2]) if present
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
        crate::processor::validate_opset(opset, 11)?;

        // TODO: Add validation for input count (1-4 inputs as per spec)
        // TODO: Add validation for output count (should be exactly 1)
        // TODO: Validate that mode attribute if present is in ["constant", "reflect", "edge"]
        // (currently only checked in extract_config)

        // Output has same type as input
        if let Some(input) = node.inputs.first() {
            node.outputs[0].ty = input.ty.clone();
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Helper function to get pads
        fn get_pads(node: &Node) -> Result<PadInput, ProcessError> {
            crate::processor::validate_min_inputs(node, 1)?;
            if node.inputs.len() >= 4 {
                return Err(ProcessError::Custom(
                    "Pad: axes input is not supported".to_string(),
                ));
            }

            let input_dim = match &node.inputs.first().unwrap().ty {
                ArgType::Tensor(tensor) => tensor.rank,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor".to_string(),
                        actual: "Pad: Only tensor input is valid".to_string(),
                    });
                }
            };

            // Check for mode attribute
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "mode" {
                    let mode = value.clone().into_string();
                    if mode != "constant" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: format!(
                                "only constant mode is supported, given mode is {}",
                                mode
                            ),
                        });
                    }
                }
            }

            // Check for pads attribute first (takes precedence)
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "pads" {
                    let pads = value
                        .clone()
                        .into_i64s()
                        .iter()
                        .map(|&x| {
                            if x < 0 {
                                return Err(ProcessError::InvalidAttribute {
                                    name: "pads".to_string(),
                                    reason: "Negative pad is not supported".to_string(),
                                });
                            }
                            Ok(x as usize)
                        })
                        .collect::<Result<Vec<usize>, ProcessError>>()?;

                    if pads.len() != input_dim * 2 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "pads".to_string(),
                            reason: "pads should be a 1D tensor of shape [2 * num_axes]"
                                .to_string(),
                        });
                    }
                    if input_dim < 2 {
                        return Err(ProcessError::Custom(
                            "Pad: input tensor should be rank 2 or higher".to_string(),
                        ));
                    }

                    // Validate and reorder pads
                    let validated_pads = validate_and_reorder_pads(&pads, input_dim)?;
                    return Ok(PadInput::Static(validated_pads));
                }
            }

            // Check for pads input
            if node.inputs.len() > 1 {
                let input = &node.inputs[1];
                match input.value() {
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        return Ok(PadInput::Runtime(RuntimeInputRef::new(
                            input.name.clone(),
                            1,
                        )));
                    }
                    Some(tensor_data) => {
                        let pads = tensor_data
                            .data
                            .into_i64s()
                            .iter()
                            .map(|&x| {
                                if x < 0 {
                                    return Err(ProcessError::Custom(
                                        "Pad: Negative pad is not supported".to_string(),
                                    ));
                                }
                                Ok(x as usize)
                            })
                            .collect::<Result<Vec<usize>, ProcessError>>()?;

                        if pads.len() != input_dim * 2 {
                            return Err(ProcessError::Custom(
                                "Pad: pads should be a 1D tensor of shape [2 * num_axes]"
                                    .to_string(),
                            ));
                        }
                        if input_dim < 2 {
                            return Err(ProcessError::Custom(
                                "Pad: input tensor should be rank 2 or higher".to_string(),
                            ));
                        }

                        // Validate and reorder pads
                        let validated_pads = validate_and_reorder_pads(&pads, input_dim)?;
                        return Ok(PadInput::Static(validated_pads));
                    }
                }
            }

            Err(ProcessError::Custom(
                "Pad: pads should be given as attribute or as input".to_string(),
            ))
        }

        fn validate_and_reorder_pads(
            pads: &[usize],
            input_dim: usize,
        ) -> Result<Vec<usize>, ProcessError> {
            let left_index = input_dim - 1;
            let top_index = input_dim - 2;
            let right_index = pads.len() - 1;
            let bottom_index = pads.len() - 2;
            let index_list = [left_index, top_index, right_index, bottom_index];

            for (index, &item) in pads.iter().enumerate() {
                if !index_list.contains(&index) && item != 0 {
                    return Err(ProcessError::Custom(
                        "Pad: padding will only be applied to the last two dimensions but found non zero padding for other dimensions".to_string(),
                    ));
                }
            }

            let left = pads[left_index];
            let top = pads[top_index];
            let right = pads[right_index];
            let bottom = pads[bottom_index];
            Ok(vec![left, right, top, bottom])
        }

        fn get_constant_value(node: &Node) -> Result<ConstantValueInput, ProcessError> {
            // Check for value attribute first (takes precedence)
            if node.attrs.contains_key("value") {
                let constant_value = node.attrs.get("value").map(|value| match value {
                    AttributeValue::Float32(value) => Ok(*value),
                    _ => Err(ProcessError::InvalidAttribute {
                        name: "value".to_string(),
                        reason: "only float32 values are currently supported for constant value as attribute".to_string(),
                    }),
                }).transpose()?.ok_or_else(|| ProcessError::Custom("constant_value should have had a value".to_string()))?;
                return Ok(ConstantValueInput::Static(constant_value));
            }

            // Check for constant value input
            if let Some(input) = node.inputs.get(2) {
                match input.value() {
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        return Ok(ConstantValueInput::Runtime(RuntimeInputRef::new(
                            input.name.clone(),
                            2,
                        )));
                    }
                    Some(tensor_data) => {
                        // TODO: Support int, boolean
                        let constant_value = match &tensor_data.data {
                            Data::Float16s(values) => values.first().map(|&f| f32::from(f)),
                            Data::Float32s(values) => values.first().copied(),
                            Data::Float64s(values) => values.first().map(|&f| f as f32),
                            Data::Float16(value) => Some(f32::from(*value)),
                            Data::Float32(value) => Some(*value),
                            Data::Float64(value) => Some(*value as f32),
                            _ => {
                                return Err(ProcessError::TypeMismatch {
                                    expected: "float value".to_string(),
                                    actual: "only float values are currently supported for constant value".to_string(),
                                });
                            }
                        };
                        return Ok(ConstantValueInput::Static(constant_value.unwrap_or(0.0)));
                    }
                }
            }

            // Default to 0.0 if no constant value provided
            Ok(ConstantValueInput::Static(0.0))
        }

        let pads = get_pads(node)?;
        let constant_value = get_constant_value(node)?;

        let config = PadConfig {
            pads,
            constant_value,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        pad_attrs: Option<Vec<i64>>,
        pad_inputs: Option<Vec<i64>>,
        constant_value_attr: Option<f32>,
        constant_value_input: Option<f32>,
        mode: Option<&str>,
        rank: usize,
    ) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("output", rank, None);

        // Add pad inputs if provided
        if let Some(pads) = pad_inputs.clone() {
            let pads_len = pads.len();
            builder = builder.input_tensor_i64_data("pads", pads, vec![pads_len]);
        }

        // Add constant value input if provided
        if let Some(value) = constant_value_input {
            builder = builder.input_scalar_tensor_f32("constant_value", Some(value));
        }

        // Add attributes if provided
        if let Some(pads) = pad_attrs {
            builder = builder.attr_ints("pads", pads);
        }

        if let Some(value) = constant_value_attr {
            builder = builder.attr_float("value", value);
        }

        if let Some(mode_val) = mode {
            builder = builder.attr_string("mode", mode_val);
        }

        builder
    }

    #[test]
    fn test_pad_config_with_attrs() {
        // Test for 2D tensor (rank 2)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.0),
            None,
            Some("constant"),
            2,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_with_inputs() {
        // For a 2D tensor, pads should have 4 values (2*rank)
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(None, Some(pads.clone()), None, Some(1.0), None, 2)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_with_3d_tensor() {
        // For a 3D tensor, pads should have 6 values (2*rank)
        let pads = vec![0, 0, 0, 0, 1, 1];
        let node = create_test_node(
            Some(pads.clone()),
            None,
            Some(0.5),
            None,
            Some("constant"),
            3,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_attrs_override_inputs() {
        // Attributes should override inputs
        let attr_pads = vec![0, 0, 2, 2];
        let input_pads = vec![0, 0, 1, 1];
        let node = create_test_node(
            Some(attr_pads.clone()),
            Some(input_pads),
            Some(0.0),
            Some(1.0),
            Some("constant"),
            2,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 2, 0, 2]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    fn create_test_node_with_runtime_inputs() -> NodeBuilder {
        NodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("pads", 1, None) // Runtime input - no static value
            .input_tensor_f32("constant_value", 0, None) // Runtime input - no static value
            .output_tensor_f32("output", 2, None)
    }

    #[test]
    fn test_pad_config_with_runtime_inputs() {
        let node = create_test_node_with_runtime_inputs().build();
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();

        // Check that we have runtime inputs
        assert!(matches!(&config.pads, PadInput::Runtime(arg) if arg.name == "pads"));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_static_runtime_pads() {
        // Static pads, runtime constant_value
        let builder = NodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("pads", vec![0, 0, 1, 1], vec![4]) // Static
            .input_tensor_f32("constant_value", 0, None) // Runtime
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();

        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_runtime_static_constant() {
        // Runtime pads, static constant_value
        let builder = NodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("pads", 1, None) // Runtime
            .input_scalar_tensor_f32("constant_value", Some(2.5)) // Static
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();

        assert!(matches!(&config.pads, PadInput::Runtime(arg) if arg.name == "pads"));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 2.5).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_default_constant_value() {
        // Test that constant_value defaults to 0.0 when not provided
        let pads = vec![0, 0, 1, 1];
        let node = create_test_node(None, Some(pads.clone()), None, None, None, 2)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<PadConfig>();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_no_inputs() {
        let mut node = create_test_node(None, None, None, None, None, 2).build_with_graph_data(16);
        node.inputs = vec![];
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_pad_config_invalid_input_type() {
        let mut node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_pad_config_with_axes_input() {
        // Create node with 4 inputs (including axes)
        let mut node = create_test_node(None, Some(vec![0, 0, 1, 1]), None, Some(0.0), None, 2)
            .build_with_graph_data(16);
        node.inputs.push(Argument {
            name: "axes".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 1,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_pad_config_negative_pad() {
        let node = create_test_node(Some(vec![0, 0, -1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_unsupported_mode() {
        let node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, Some("reflect"), 2)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_no_pads() {
        let node = create_test_node(None, None, None, None, None, 2).build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_pad_config_invalid_pads_length() {
        let node = create_test_node(Some(vec![0, 0, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_pad_config_invalid_tensor_rank() {
        let node =
            create_test_node(Some(vec![0, 1]), None, None, None, None, 1).build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_pad_config_non_zero_padding_on_other_dimensions() {
        // For a 3D tensor, we try to set non-zero padding on first dimension
        let node = create_test_node(Some(vec![1, 0, 0, 0, 1, 1]), None, None, None, None, 3)
            .build_with_graph_data(16);
        let node = node;
        let processor = PadProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
