use crate::ir::Node;

/// Configuration for `ConvTranspose2d` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    /// Input and output channels [in, out].
    pub channels: [usize; 2],
    /// Size of the kernel.
    pub kernel_size: [usize; 2],
    /// Stride of the convolutional kernel.
    pub stride: [usize; 2],
    /// Dilation of the convolutional kernel.
    pub dilation: [usize; 2],
    /// Padding.
    pub padding: [usize; 2],
    /// Output padding.
    pub padding_out: [usize; 2],
    /// Groups.
    pub groups: usize,
    /// Use bias.
    pub bias: bool,
}

impl ConvTranspose2dConfig {
    /// Create a new configuration for a `ConvTranspose2d`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        padding: [usize; 2],
        padding_out: [usize; 2],
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

/// Create a `ConvTranspose2dConfig` from the attributes of the node
#[must_use]
pub fn conv_transpose2d_config(curr: &Node) -> ConvTranspose2dConfig {
    let mut kernel_shape = Vec::new(); // Default to empty vector
    let mut stride = vec![1, 1]; // Default stride to 1
    let mut pads = vec![0, 0, 0, 0]; // Default padding to 0
    let mut dilations = vec![1, 1]; // Default dilation to 1
    let mut group: usize = 1; // Default group to 1
    let mut output_padding = vec![0, 0]; // Default output padding to 0

    // Extract attributes
    for (key, value) in &curr.attrs {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => stride = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
            "group" => group = value.clone().into_i64() as usize,
            "output_padding" => output_padding = value.clone().into_i64s(),
            _ => panic!("Unexpected attribute for ConvTranspose2d: {key}"),
        }
    }

    // Check the pads are symmetric.
    let [left, top, right, bottom] = [pads[0], pads[1], pads[2], pads[3]];
    if left < 0 || top < 0 || right < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) {
        panic!("Asymmetric padding is not supported");
    }

    let weight_shape = curr.inputs[1]
        .value
        .as_ref()
        .expect("ConvTranspose2d: weight tensor must be present")
        .shape
        .clone();

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let channels: [usize; 2] = [weight_shape[1] * group, weight_shape[0]];

    ConvTranspose2dConfig::new(
        channels,
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
        [stride[0] as usize, stride[1] as usize],
        [dilations[0] as usize, dilations[1] as usize],
        [pads[0] as usize, pads[1] as usize],
        [output_padding[0] as usize, output_padding[1] as usize],
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
        output_padding: Vec<i64>,
        group: i64,
        has_bias: bool,
    ) -> Node {
        // Create weight tensor data
        let weight_data = vec![0.0; 16]; // Not important for the test

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::ConvTranspose2d, "test_convtranspose2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data(
                "weight",
                weight_data,
                vec![2, 4, 2, 2], // [out_channels, in_channels, k_h, k_w]
            )
            .output_tensor_f32("output", 4, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        // Add attributes
        builder = builder
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_ints("output_padding", output_padding)
            .attr_int("group", group);

        builder.build()
    }

    #[test]
    fn test_conv_transpose2d_config_basic() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
        );
        let config = conv_transpose2d_config(&node);

        assert_eq!(config.channels, [4, 2]);
        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.padding, [0, 0]);
        assert_eq!(config.padding_out, [0, 0]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    fn test_conv_transpose2d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3],
            vec![2, 2],
            vec![1, 1, 1, 1],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
        );
        let config = conv_transpose2d_config(&node);

        assert_eq!(config.padding, [1, 1]);
        assert_eq!(config.stride, [2, 2]);
    }

    #[test]
    fn test_conv_transpose2d_config_with_output_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![2, 2],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![1, 1],
            1,
            false,
        );
        let config = conv_transpose2d_config(&node);

        assert_eq!(config.padding_out, [1, 1]);
    }

    #[test]
    fn test_conv_transpose2d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            2,
            false,
        );
        let config = conv_transpose2d_config(&node);

        assert_eq!(config.groups, 2);
        assert_eq!(config.channels, [8, 2]); // channels_in is adjusted by groups
    }

    #[test]
    fn test_conv_transpose2d_config_with_bias() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            true,
        );
        let config = conv_transpose2d_config(&node);

        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_conv_transpose2d_config_with_asymmetric_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![1, 1, 2, 2],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
        );
        let _ = conv_transpose2d_config(&node);
    }
}
