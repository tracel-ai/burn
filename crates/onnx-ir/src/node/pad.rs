//! # Pad
//!
//! Pads input tensor with additional values at borders.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Pad.html>
//!
//! ## Opset Versions
//! - **Opset 11**: Changed pads from attribute to input for dynamic padding support. Added mode attribute (constant/reflect/edge).
//! - **Opset 13**: Added optional axes input to specify which axes to pad (not supported in this implementation).
//! - **Opset 18**: Added optional constant_value input as alternative to attribute.
//! - **Opset 19**: Added antialiasing support for edge mode (not supported in this implementation).
//!
//! **Implementation Note**: This implementation requires opset 11+ and only supports constant mode padding. The axes input (opset 13+) is explicitly rejected.
//!
//! FIXME: Implementation only supports padding on the last 2 dimensions
//! The validate_and_reorder_pads function (lines 286-309) enforces that only the last two dimensions
//! can have non-zero padding values. This is a major spec deviation - ONNX allows padding any dimension.
//! This limitation should either be:
//! 1. Fixed to support arbitrary dimension padding per spec, OR
//! 2. Clearly documented as a known limitation with validation moved to infer_types
//!    Impact: HIGH - Models padding batch/channel dimensions will fail with cryptic error messages.
//!
//! TODO: Missing type constraint validation
//! Spec defines type constraints for T (data/output), but implementation doesn't validate.
//! Should validate constant_value type matches data type when provided.
//! Location: extract_config or infer_types

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{
    ArgType, AttributeValue, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt,
};
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

/// Padding mode for Pad operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Constant padding (fill with constant value).
    #[default]
    Constant,
    /// Reflect padding (mirror values).
    Reflect,
    /// Edge padding (replicate edge values).
    Edge,
}

impl std::str::FromStr for PadMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "constant" => Ok(PadMode::Constant),
            "reflect" => Ok(PadMode::Reflect),
            "edge" => Ok(PadMode::Edge),
            _ => Err(format!("Invalid pad mode: {}", s)),
        }
    }
}

impl PadMode {
    /// Convert PadMode to string for serialization.
    pub fn as_str(&self) -> &str {
        match self {
            PadMode::Constant => "constant",
            PadMode::Reflect => "reflect",
            PadMode::Edge => "edge",
        }
    }
}

/// Configuration for the Pad operation.
#[derive(Debug, Clone)]
pub struct PadConfig {
    /// The paddings to be applied to each dimension.
    pub pads: PadInput,
    /// The constant value to fill the padded areas with.
    pub constant_value: ConstantValueInput,
    /// The padding mode (constant, reflect, edge). Default: constant.
    pub mode: PadMode,
}

impl Default for PadConfig {
    fn default() -> Self {
        Self {
            pads: PadInput::Static(vec![]),
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::default(),
        }
    }
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
    type Config = PadConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Range(1, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    // TODO mark axes inputs as Shape if inputs are constant

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for input count (1-4 inputs as per spec)
        // Spec allows 1-4 inputs (data, pads, constant_value optional, axes optional).
        // Currently no explicit validation, though extract_config rejects 4 inputs (axes).
        // Should add: validate_min_inputs(node, 1) and validate_max_inputs(node, 4)
        // Location: After validate_opset

        // TODO: Add validation for output count (should be exactly 1)
        // Missing explicit output count validation. Should add validate_output_count(node, 1).
        // Location: After input count validation

        // TODO: Validate that mode attribute if present is in ["constant", "reflect", "edge"]
        // Mode validation currently only happens in extract_config, not in infer_types.
        // This means type inference succeeds even with invalid mode, failing later.
        // Should validate mode attribute early in infer_types for better error messages.
        // Location: After output count validation

        // TODO: Missing test coverage for reflect and edge padding modes
        // Implementation defines PadMode::Reflect and PadMode::Edge but explicitly rejects them
        // in extract_config (line 167-174). Either implement these modes or remove enum variants.
        // Tests only cover constant mode padding. Add tests: pad_reflect_mode, pad_edge_mode
        // (expect error with current implementation)

        // TODO: Missing test coverage for different data types
        // Tests only use f32 tensors. Spec supports all numeric types including int8, int16, etc.
        // Add tests: pad_int32, pad_int64, pad_float64, pad_bool

        // TODO: Missing test coverage for 1D and 4D+ tensors
        // Tests cover 2D and 3D tensors, but implementation requires rank >= 2 (line 225-229, 268-271).
        // This contradicts ONNX spec which allows 1D tensors. Either fix implementation or clarify.
        // Add tests: pad_1d_tensor (should work per spec), pad_4d_tensor, pad_5d_tensor

        // FIXME: Implementation restricts padding to last 2 dimensions only
        // Lines 286-302 in validate_and_reorder_pads enforce that only the last two dimensions
        // can have non-zero padding. This is a significant deviation from ONNX spec which allows
        // padding any dimension. This should be documented prominently or fixed.
        // Impact: Models using batch/channel padding will fail unexpectedly.

        // Output has same type as input
        if let Some(input) = node.inputs.first() {
            node.outputs[0].ty = input.ty.clone();
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Helper function to get mode
        fn get_mode(node: &NodeBuilder) -> Result<PadMode, ProcessError> {
            use std::str::FromStr;

            // Check for mode attribute (default is "constant")
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "mode" {
                    let mode_str = value.clone().into_string();
                    let mode = PadMode::from_str(&mode_str).map_err(|e| {
                        ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: e,
                        }
                    })?;

                    // Current implementation only supports constant mode
                    if mode != PadMode::Constant {
                        return Err(ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: format!(
                                "only constant mode is supported, given mode is {}",
                                mode.as_str()
                            ),
                        });
                    }
                    return Ok(mode);
                }
            }
            Ok(PadMode::default())
        }

        fn get_pads(node: &NodeBuilder) -> Result<PadInput, ProcessError> {
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
                        let pad_values: Vec<i64> = tensor_data.to_vec().unwrap();
                        let pads = pad_values
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

        fn get_constant_value(node: &NodeBuilder) -> Result<ConstantValueInput, ProcessError> {
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
                        // Static input - extract the scalar value, converting to f32
                        match tensor_data.scalar_f32() {
                            Ok(value) => return Ok(ConstantValueInput::Static(value)),
                            Err(_) => {
                                return Err(ProcessError::TypeMismatch {
                                    expected: "float value".to_string(),
                                    actual: "only float values are currently supported for constant value".to_string(),
                                });
                            }
                        }
                    }
                }
            }

            // Default to 0.0 if no constant value provided
            Ok(ConstantValueInput::Static(0.0))
        }

        let mode = get_mode(node)?;
        let pads = get_pads(node)?;
        let constant_value = get_constant_value(node)?;

        let config = PadConfig {
            pads,
            constant_value,
            mode,
        };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Pad {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        pad_attrs: Option<Vec<i64>>,
        pad_inputs: Option<Vec<i64>>,
        constant_value_attr: Option<f32>,
        constant_value_input: Option<f32>,
        mode: Option<&str>,
        rank: usize,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
        assert_eq!(config.mode, PadMode::Constant);
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 2, 0, 2]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    fn create_test_node_with_runtime_inputs() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Pad, "test_pad")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check that we have runtime inputs
        assert!(matches!(&config.pads, PadInput::Runtime(arg) if arg.name == "pads"));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_static_runtime_pads() {
        // Static pads, runtime constant_value
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("pads", vec![0, 0, 1, 1], vec![4]) // Static
            .input_tensor_f32("constant_value", 0, None) // Runtime
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Runtime(arg) if arg.name == "constant_value")
        );
    }

    #[test]
    fn test_pad_config_mixed_runtime_static_constant() {
        // Runtime pads, static constant_value
        let builder = TestNodeBuilder::new(NodeType::Pad, "test_pad")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("pads", 1, None) // Runtime
            .input_scalar_tensor_f32("constant_value", Some(2.5)) // Static
            .output_tensor_f32("output", 2, None);

        let node = builder.build_with_graph_data(16);
        let mut node = node;
        let processor = PadProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(&config.pads, PadInput::Static(pads) if pads == &vec![0, 1, 0, 1]));
        assert!(
            matches!(&config.constant_value, ConstantValueInput::Static(v) if (*v - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_pad_config_no_inputs() {
        let mut node = create_test_node(None, None, None, None, None, 2).build_with_graph_data(16);
        node.inputs = vec![];
        let processor = PadProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }

    #[test]
    fn test_pad_config_invalid_input_type() {
        let mut node = create_test_node(Some(vec![0, 0, 1, 1]), None, None, None, None, 2)
            .build_with_graph_data(16);
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
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
                dtype: DType::I64,
                rank: 1,
                static_shape: None,
            }),
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
