use crate::ir::{Node, NodeConfig};
use crate::util::validate_opset;

use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::NodeProcessor;
use crate::util::same_as_input;
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
    fn process_config(&self, node: &mut Node, opset: usize) {
        // Conv implementation supports opset 1+
        validate_opset(&node.node_type, opset, 1);

        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut group: usize = 1;

        let weight_shape = node.inputs[1]
            .into_value()
            .expect("Conv2d: weight tensor must be present")
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
                        panic!("Unsupported 'auto_pad' value: {auto_pad}");
                    }
                }
                _ => panic!("Unexpected attribute for Conv2d: {key}"),
            }
        }

        // the channels are inverted in the weight tensor
        let channels_in = weight_shape[1] * group;
        let channels_out = weight_shape[0];

        let padding = padding_config_2d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            // https://onnx.ai/onnx/operators/onnx__Conv.html#attributes
            // Spec says if kernel shape not present in attributes it should be inferred from
            // the weight tensor, which has shape (M, C/group, kH, kW).
            if weight_shape.len() != 4 {
                panic!(
                    "expected to infer kernel shape from a weight tensor of rank 4 but got shape {weight_shape:?}"
                );
            }

            [weight_shape[2], weight_shape[3]]
        } else {
            // Was set explicitly via attributes- use that
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

        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        // Conv2d preserves rank (same as input)
        same_as_input(node);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.kernel_size, [3, 3]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    #[should_panic = "Unsupported 'auto_pad' value"]
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
        let mut node = node;
        let processor = Conv2dProcessor;
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
        let config = node.config::<Conv2dConfig>();

        assert_eq!(config.kernel_size, [2, 2]); // Inferred via weight tensor shape
    }
}
