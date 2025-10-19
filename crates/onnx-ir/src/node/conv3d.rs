use crate::ir::{Node, NodeConfig};

use crate::node::padding::{PaddingConfig3d, padding_config_3d};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for Conv3d operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv3dConfig {
    /// Input and output channels [in, out].
    pub channels: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 3],
    /// Stride of the convolutional kernel.
    pub stride: [usize; 3],
    /// Dilation of the convolutional kernel.
    pub dilation: [usize; 3],
    /// Groups.
    pub groups: usize,
    /// Use bias.
    pub bias: bool,
    /// Padding.
    pub padding: PaddingConfig3d,
}

impl Conv3dConfig {
    /// Create a new configuration for a Conv3d.
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
        bias: bool,
        padding: PaddingConfig3d,
    ) -> Self {
        Self {
            channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            padding,
        }
    }
}

impl NodeConfig for Conv3dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct Conv3dProcessor;

impl NodeProcessor for Conv3dProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {

        // Lift weight (input[1]) and optional bias (input[2])
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 {
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
        crate::util::validate_opset(opset, 1)?;
        crate::util::validate_min_inputs(node, 2)?;
        crate::util::validate_output_count(node, 1)?;

        // Output type is same as input
        crate::util::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1, 1];
        let mut pads = vec![0, 0, 0, 0, 0, 0];
        let mut dilations = vec![1, 1, 1];
        let mut group: usize = 1;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Conv3d: weight tensor must be present".to_string())
            })?
            .shape
            .clone();

        // check if the bias is present
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
                            reason: format!("Unsupported 'auto_pad' value: {auto_pad}"),
                        });
                    }
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Conv3d: {key}"),
                    });
                }
            }
        }

        // the channels are inverted in the weight tensor
        let channels_in = weight_shape[1] * group;
        let channels_out = weight_shape[0];

        let padding = padding_config_3d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
            if weight_shape.len() != 5 {
                return Err(ProcessError::Custom(format!(
                    "expected to infer kernel shape from a weight tensor of rank 5 but got shape {weight_shape:?}"
                )));
            }

            [weight_shape[2], weight_shape[3], weight_shape[4]]
        } else {
            [
                kernel_shape[0] as _,
                kernel_shape[1] as _,
                kernel_shape[2] as _,
            ]
        };

        let config = Conv3dConfig::new(
            [channels_in, channels_out],
            kernel_size,
            [
                strides[0] as usize,
                strides[1] as usize,
                strides[2] as usize,
            ],
            [
                dilations[0] as usize,
                dilations[1] as usize,
                dilations[2] as usize,
            ],
            group,
            bias,
            padding,
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
        // Create weight tensor data (not important for the test)
        let weight_data = vec![0.0; 32];
        let weight_shape = vec![4, 2, 2, 2, 2]; // [output_channels, input_channels/groups, k_d, k_h, k_w]

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::Conv3d, "test_conv3d")
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
    fn test_conv3d_config_basic() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert_eq!(config.channels, [2, 4]);
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_conv3d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![1, 1, 1, 1, 1, 1],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert_eq!(config.kernel_size, [3, 3, 3]);
        assert!(matches!(config.padding, PaddingConfig3d::Explicit(1, 1, 1)));
    }

    #[test]
    fn test_conv3d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert_eq!(config.groups, 2);
        assert_eq!(config.channels, [4, 4]); // channels_in is adjusted by groups
    }

    #[test]
    fn test_conv3d_config_with_bias() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            true,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert!(config.bias);
    }

    #[test]
    fn test_conv3d_config_autopad_not_set() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert_eq!(config.channels, [2, 4]);
        assert_eq!(config.kernel_size, [2, 2, 2]);
        assert_eq!(config.stride, [1, 1, 1]);
        assert_eq!(config.dilation, [1, 1, 1]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig3d::Valid));
    }

    #[test]
    fn test_conv3d_config_autopad_not_supported() {
        let node = create_test_node(
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let processor = Conv3dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_conv3d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv3dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<Conv3dConfig>();

        assert_eq!(config.kernel_size, [2, 2, 2]); // Inferred via weight tensor shape
    }
}
