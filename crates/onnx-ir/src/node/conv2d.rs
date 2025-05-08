use crate::ir::Node;
use crate::node::padding::{PaddingConfig2d, padding_config_2d};

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

/// Create a Conv2dConfig from the attributes of the node
pub fn conv2d_config(curr: &Node) -> Conv2dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut dilations = vec![1, 1];
    let mut group: usize = 1;

    let weight_shape = curr.inputs[1]
        .value
        .as_ref()
        .expect("Conv2d: weight tensor must be present")
        .shape
        .clone();

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            _ => panic!("Unexpected attribute for Conv2d: {key}"),
        }
    }

    // the channels are inverted in the weight tensor
    let channels_in = weight_shape[1] * group;
    let channels_out = weight_shape[0];

    let padding = padding_config_2d(&pads);

    Conv2dConfig::new(
        [channels_in, channels_out],
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
        [strides[0] as usize, strides[1] as usize],
        padding,
        [dilations[0] as usize, dilations[1] as usize],
        group,
        bias,
    )
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
    ) -> Node {
        // Weight tensor data - not important for the test
        let weight_data = vec![0.0; 16];
        // [output_channels, input_channels/groups, k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];

        let mut builder = NodeBuilder::new(NodeType::Conv2d, "test_conv2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data.clone(), weight_shape)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group);

        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        builder.build()
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
        );
        let config = conv2d_config(&node);

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
        );
        let config = conv2d_config(&node);

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
        );
        let config = conv2d_config(&node);

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
        );
        let config = conv2d_config(&node);

        assert!(config.bias);
    }
}
