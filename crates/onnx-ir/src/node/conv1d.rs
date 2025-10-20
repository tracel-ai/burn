//! # Conv (1D)
//!
//! 1D convolution operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Conv.html>
//!
//! ## Attributes
//! - `kernel_shape` (optional): Kernel size \[width\]
//! - `strides` (optional): Stride \[width\], default \[1\]
//! - `pads` (optional): Padding \[left, right\], default \[0, 0\]
//! - `dilations` (optional): Dilation \[width\], default \[1\]
//! - `group` (optional): Number of groups, default 1
//! - `auto_pad` (optional): Padding mode (only `NOTSET` supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x L)
//! - `W` (T): Weight tensor (M x C/group x kW)
//! - `B` (T, optional): Bias tensor (M)
//!
//! ## Outputs
//! - `Y` (T): Output tensor
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic convolution support
//! - **Opset 11**: No changes to Conv operator itself (broader ONNX updates)

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

use super::padding::{PaddingConfig1d, padding_config_1d};

/// Configuration for Conv1d operations
#[derive(Debug, Clone)]
pub struct Conv1dConfig {
    /// Input channels
    pub channels_in: usize,
    /// Output channels
    pub channels_out: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation
    pub dilation: usize,
    /// Number of groups
    pub groups: usize,
    /// Whether bias is used
    pub bias: bool,
    /// Padding configuration
    pub padding: PaddingConfig1d,
}

impl Conv1dConfig {
    /// Create a new Conv1dConfig
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: PaddingConfig1d,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        }
    }
}

impl NodeConfig for Conv1dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Node processor for Conv1d operation
pub struct Conv1dProcessor;

impl NodeProcessor for Conv1dProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift weight (input[1]) and optional bias (input[2])
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
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_min_inputs(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // Extract input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Conv1d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1];
        let mut pads = vec![0, 0];
        let mut dilations = vec![1];
        let mut group: usize = 1;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Conv1d: weight tensor must be present".to_string())
            })?
            .shape()
            .to_vec();

        let bias = node.inputs.len() == 3;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "group" => group = value.clone().into_i64() as usize,
                "auto_pad" => {
                    let auto_pad = value.clone().into_string();
                    if auto_pad != "NOTSET" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "auto_pad".to_string(),
                            reason: format!("Unsupported 'auto_pad' value: {}", auto_pad),
                        });
                    }
                }
                _ => {
                    // TODO: According to spec, there may be other valid attributes that are not handled
                    // Consider logging/warning instead of rejecting unknown attributes
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Conv1d: {}", key),
                    });
                }
            }
        }

        let channels_in = weight_shape[1] * group;
        let channels_out = weight_shape[0];

        let padding = padding_config_1d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            if weight_shape.len() != 3 {
                return Err(ProcessError::Custom(format!(
                    "Conv1d: expected to infer kernel shape from a weight tensor of rank 3 but got shape {:?}",
                    weight_shape
                )));
            }
            weight_shape[2]
        } else {
            kernel_shape[0] as _
        };

        let config = Conv1dConfig {
            channels_in,
            channels_out,
            kernel_size,
            stride: strides[0] as usize,
            dilation: dilations[0] as usize,
            groups: group,
            bias,
            padding,
        };

        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> NodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.1; 16];

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::Conv1d, "test_conv1d")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32_data(
                "weight",
                weight_data,
                vec![2, 2, 4], // [out_channels, in_channels, kernel_size]
            )
            .output_tensor_f32("output", 3, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32_data("bias", vec![0.1, 0.2], vec![2]);
        }

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        builder
    }

    #[test]
    fn test_conv1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, false, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_conv1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1], 1, true, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.groups, 1);
        assert!(config.bias);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_conv1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2], 1, false, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_conv1d_config_with_groups() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 2, false, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.channels_in, 4);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.groups, 2);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_conv1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 1, false, None)
            .build_with_graph_data(16);
        let processor = Conv1dProcessor;
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_conv1d_config_negative_padding() {
        let node = create_test_node(vec![4], vec![1], vec![-1, -1], vec![1], 1, false, None)
            .build_with_graph_data(16);
        let processor = Conv1dProcessor;
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_conv1d_config_autopad_not_set() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_conv1d_config_autopad_not_supported() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let processor = Conv1dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_conv1d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv1dConfig>();

        assert_eq!(config.kernel_size, 4); // Inferred via weight tensor shape
    }
}
