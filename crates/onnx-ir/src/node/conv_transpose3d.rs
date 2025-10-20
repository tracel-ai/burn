//! # ConvTranspose (3D)
//!
//! 3D transposed convolution (deconvolution) operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ConvTranspose.html>
//!
//! ## Attributes
//! - `kernel_shape` (optional): Kernel size \[depth, height, width\]
//! - `strides` (optional): Stride \[depth, height, width\], default \[1, 1, 1\]
//! - `pads` (optional): Padding \[d_begin, h_begin, w_begin, d_end, h_end, w_end\], default \[0, 0, 0, 0, 0, 0\] (must be symmetric)
//! - `dilations` (optional): Dilation \[depth, height, width\], default \[1, 1, 1\]
//! - `group` (optional): Number of groups, default 1
//! - `output_padding` (optional): Output padding \[depth, height, width\], default \[0, 0, 0\]
//! - `auto_pad` (optional): Padding mode (only `NOTSET` supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x D x H x W)
//! - `W` (T): Weight tensor (C x M/group x kD x kH x kW)
//! - `B` (T, optional): Bias tensor (M)
//!
//! ## Outputs
//! - `Y` (T): Output tensor
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic transposed convolution support
//! - **Opset 11**: No changes to ConvTranspose operator itself (broader ONNX updates)

use crate::ir::{Node, NodeConfig};

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for ConvTranspose3d operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvTranspose3dConfig {
    /// Input and output channels [in, out].
    pub channels: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Stride of the convolutional kernel.
    pub stride: [usize; 3],
    /// Dilation of the convolutional kernel.
    pub dilation: [usize; 3],
    /// Padding.
    pub padding: [usize; 3],
    /// Output padding.
    pub padding_out: [usize; 3],
    /// Groups.
    pub groups: usize,
    /// Use bias.
    pub bias: bool,
}

impl ConvTranspose3dConfig {
    /// Create a new configuration for a ConvTranspose3d.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        padding: [usize; 3],
        padding_out: [usize; 3],
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            channels,
            kernel_size,
            stride,
            dilation,
            padding,
            padding_out,
            groups,
            bias,
        }
    }
}

impl NodeConfig for ConvTranspose3dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct Convtranspose3dProcessor;

impl NodeProcessor for Convtranspose3dProcessor {
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

        // Output type inference
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1, 1, 1]; // Default stride to 1
        let mut pads = vec![0, 0, 0, 0, 0, 0]; // Default padding to 0
        let mut dilations = vec![1, 1, 1]; // Default dilation to 1
        let mut group: usize = 1; // Default group to 1
        let mut output_padding = vec![0, 0, 0]; // Default output padding to 0

        // Extract attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => stride = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "group" => group = value.clone().into_i64() as usize,
                "output_padding" => output_padding = value.clone().into_i64s(),
                "auto_pad" => {
                    let auto_pad = value.clone().into_string();
                    if auto_pad != "NOTSET" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "auto_pad".to_string(),
                            reason: format!("Unsupported 'auto_pad' value: {auto_pad}"),
                        });
                    }
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for ConvTranspose3d: {key}"),
                    });
                }
            }
        }

        // Check the pads are symmetric.
        let [left, top, front, right, bottom, back] =
            [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

        if left < 0 || top < 0 || front < 0 || right < 0 || bottom < 0 || back < 0 {
            return Err(ProcessError::Custom(
                "Negative pad values are not supported".to_string(),
            ));
        } else if (left != right) || (top != bottom) || (front != back) {
            return Err(ProcessError::Custom(
                "Asymmetric padding is not supported".to_string(),
            ));
        }

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("ConvTranspose3d: weight tensor must be present".to_string())
            })?
            .shape()
            .to_vec();

        // check if the bias is present
        let bias = node.inputs.len() == 3;

        // the channels are inverted in the weight tensor
        let channels: [usize; 2] = [weight_shape[1] * group, weight_shape[0]];

        let kernel_size = if kernel_shape.is_empty() {
            // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
            // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
            if weight_shape.len() != 5 {
                return Err(ProcessError::Custom(format!(
                    "expected to infer kernel shape from a weight tensor of rank 5 but got shape {weight_shape:?}"
                )));
            }

            [weight_shape[2], weight_shape[3], weight_shape[4]]
        } else {
            // Was set explicitly via attributes- use that
            [
                kernel_shape[0] as _,
                kernel_shape[1] as _,
                kernel_shape[2] as _,
            ]
        };

        let config = ConvTranspose3dConfig::new(
            channels,
            kernel_size,
            [stride[0] as usize, stride[1] as usize, stride[2] as usize],
            [
                dilations[0] as usize,
                dilations[1] as usize,
                dilations[2] as usize,
            ],
            [pads[0] as usize, pads[1] as usize, pads[2] as usize],
            [
                output_padding[0] as usize,
                output_padding[1] as usize,
                output_padding[2] as usize,
            ],
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

    #[allow(clippy::too_many_arguments)]
    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        output_padding: Vec<i64>,
        group: i64,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> NodeBuilder {
        // Create weight tensor data
        let weight_shape = vec![2, 4, 2, 2, 2]; // [out_channels, in_channels, k_d, k_h, k_w]
        let weight_data = vec![0.0; 64]; // 2*4*2*2*2 = 64

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::ConvTranspose3d, "test_convtranspose3d")
            .input_tensor_f32("data", 5, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .output_tensor_f32("output", 5, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_ints("output_padding", output_padding)
            .attr_int("group", group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        builder
    }

    #[test]
    fn test_conv_transpose3d_config_basic() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.channels, [4, 2]);
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.padding, [0, 0, 0]);
        assert_eq!(config.padding_out, [0, 0, 0]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    fn test_conv_transpose3d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3, 3],
            vec![2, 2, 2],
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.padding, [1, 1, 1]);
        assert_eq!(config.stride, [2, 2, 2]);
    }

    #[test]
    fn test_conv_transpose3d_config_with_output_padding() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![2, 2, 2],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.padding_out, [1, 1, 1]);
    }

    #[test]
    fn test_conv_transpose3d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.groups, 2);
        assert_eq!(config.channels, [8, 2]); // channels_in is adjusted by groups
    }

    #[test]
    fn test_conv_transpose3d_config_with_bias() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            true,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert!(config.bias);
    }

    #[test]
    fn test_conv_transpose3d_config_with_asymmetric_padding() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let node = node;
        let processor = Convtranspose3dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_err());
        match result {
            Err(ProcessError::Custom(msg)) => {
                assert!(msg.contains("Asymmetric padding is not supported"));
            }
            _ => panic!("Expected ProcessError::Custom with asymmetric padding message"),
        }
    }

    #[test]
    fn test_conv_transpose3d_config_autopad_not_set() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.channels, [4, 2]);
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.padding, [0, 0, 0]);
        assert_eq!(config.padding_out, [0, 0, 0]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    fn test_conv_transpose3d_config_autopad_not_supported() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let node = node;
        let processor = Convtranspose3dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(result.is_err());
        match result {
            Err(ProcessError::InvalidAttribute { .. }) => {}
            _ => panic!("Expected ProcessError::InvalidAttribute"),
        }
    }

    #[test]
    fn test_conv3d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            vec![0, 0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ConvTranspose3dConfig>();

        assert_eq!(config.kernel_size, [2, 2, 2]); // Inferred via weight tensor shape
    }
}
