use crate::ir::Node;

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

/// Create a ConvTranspose3dConfig from the attributes of the node
pub fn conv_transpose3d_config(curr: &Node) -> ConvTranspose3dConfig {
    let mut kernel_shape = Vec::new();
    let mut stride = vec![1, 1, 1]; // Default stride to 1
    let mut pads = vec![0, 0, 0, 0, 0, 0]; // Default padding to 0
    let mut dilations = vec![1, 1, 1]; // Default dilation to 1
    let mut group: usize = 1; // Default group to 1
    let mut output_padding = vec![0, 0, 0]; // Default output padding to 0

    // Extract attributes
    for (key, value) in curr.attrs.iter() {
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
                    panic!("Unsupported 'auto_pad' value: {auto_pad}");
                }
            }
            _ => panic!("Unexpected attribute for ConvTranspose3d: {key}"),
        }
    }

    // Check the pads are symmetric.
    let [left, top, front, right, bottom, back] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || top < 0 || front < 0 || right < 0 || bottom < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if (left != right) || (top != bottom) || (front != back) {
        panic!("Asymmetric padding is not supported");
    }

    let weight_shape = curr.inputs[1]
        .value
        .as_ref()
        .expect("ConvTranspose3d: weight tensor must be present")
        .shape
        .clone();

    // check if the bias is present
    let bias = curr.inputs.len() == 3;

    // the channels are inverted in the weight tensor
    let channels: [usize; 2] = [weight_shape[1] * group, weight_shape[0]];

    let kernel_size = if kernel_shape.is_empty() {
        // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
        // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
        if weight_shape.len() != 5 {
            panic!(
                "expected to infer kernel shape from a weight tensor of rank 5 but got shape {weight_shape:?}"
            );
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

    ConvTranspose3dConfig::new(
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
    )
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
    ) -> Node {
        // Create weight tensor data
        let weight_data = vec![0.0; 32]; // Not important for the test

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::ConvTranspose3d, "test_convtranspose3d")
            .input_tensor_f32("data", 5, None)
            .input_tensor_f32_data(
                "weight",
                weight_data,
                vec![2, 4, 2, 2, 2], // [out_channels, in_channels, k_d, k_h, k_w]
            )
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

        builder.build()
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
        );
        let config = conv_transpose3d_config(&node);

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
        );
        let config = conv_transpose3d_config(&node);

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
        );
        let config = conv_transpose3d_config(&node);

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
        );
        let config = conv_transpose3d_config(&node);

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
        );
        let config = conv_transpose3d_config(&node);

        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
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
        );
        let _ = conv_transpose3d_config(&node);
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
        );
        let config = conv_transpose3d_config(&node);

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
    #[should_panic = "Unsupported 'auto_pad' value"]
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
        );
        let _config = conv_transpose3d_config(&node);
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
        );
        let config = conv_transpose3d_config(&node);

        assert_eq!(config.kernel_size, [2, 2, 2]); // Inferred via weight tensor shape
    }
}
