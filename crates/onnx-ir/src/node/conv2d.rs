//! # Conv (2D)
//!
//! 2D convolution operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Conv.html>
//!
//! ## Attributes
//! - `kernel_shape` (optional): Kernel size \[height, width\]
//! - `strides` (optional): Stride \[height, width\], default \[1, 1\]
//! - `pads` (optional): Padding \[top, left, bottom, right\], default \[0, 0, 0, 0\]
//! - `dilations` (optional): Dilation \[height, width\], default \[1, 1\]
//! - `group` (optional): Number of groups, default 1
//! - `auto_pad` (optional): Padding mode (only `NOTSET` supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x H x W)
//! - `W` (T): Weight tensor (M x C/group x kH x kW)
//! - `B` (T, optional): Bias tensor (M)
//!
//! ## Outputs
//! - `Y` (T): Output tensor
//!
//! ## Opset Versions
//! - Opset 1+

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for Conv2d operations
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Channels [in, out]
    pub channels: [usize; 2],
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub stride: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width]
    pub dilation: [usize; 2],
    /// Number of groups
    pub groups: usize,
    /// Whether bias is used
    pub bias: bool,
}

impl Conv2dConfig {
    /// Create a new Conv2dConfig
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: PaddingConfig2d,
        dilation: [usize; 2],
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        }
    }
}

impl NodeConfig for Conv2dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Node processor for Conv2d operation
pub struct Conv2dProcessor;

impl NodeProcessor for Conv2dProcessor {
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
        // Validate opset
        crate::processor::validate_opset(opset, 1)?;

        // Validate input and output counts
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

        // Conv2d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
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
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut group: usize = 1;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Conv2d: weight tensor must be present".to_string())
            })?
            .shape
            .clone();

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
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Conv2d: {}", key),
                    });
                }
            }
        }

        let channels_in = weight_shape[1] * group;
        let channels_out = weight_shape[0];

        let padding = padding_config_2d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            if weight_shape.len() != 4 {
                return Err(ProcessError::Custom(format!(
                    "Conv2d: expected to infer kernel shape from a weight tensor of rank 4 but got shape {:?}",
                    weight_shape
                )));
            }
            [weight_shape[2], weight_shape[3]]
        } else {
            [kernel_shape[0] as _, kernel_shape[1] as _]
        };

        let config = Conv2dConfig::new(
            [channels_in, channels_out],
            kernel_size,
            [strides[0] as usize, strides[1] as usize],
            padding,
            [dilations[0] as usize, dilations[1] as usize],
            group,
            bias,
        );

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
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [output_channels, input_channels/groups, k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];

        let has_kernel_shape = !kernel_shape.is_empty();

        let mut builder = NodeBuilder::new(NodeType::Conv2d, "test_conv2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data.clone(), weight_shape)
            .output_tensor_f32("output", 4, None)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        builder
    }

    #[test]
    fn test_conv2d_config_basic() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.channels, [2, 4]);
        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_conv2d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.kernel_size, [3, 3]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_conv2d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.groups, 2);
        assert_eq!(config.channels, [4, 4]); // channels_in is adjusted by groups
    }

    #[test]
    fn test_conv2d_config_with_bias() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            true,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert!(config.bias);
    }

    #[test]
    fn test_conv2d_config_autopad_not_set() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.kernel_size, [3, 3]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_conv2d_config_autopad_not_supported() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let processor = Conv2dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_conv2d_config_kernel_shape_not_set() {
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
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.kernel_size, [2, 2]); // Inferred via weight tensor shape
    }
}
