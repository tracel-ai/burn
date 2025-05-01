use crate::ir::{AttributeValue, Node};

/// Configuration for ConvTranspose2d operations.
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
    /// Create a new configuration for a ConvTranspose2d.
    #[allow(clippy::too_many_arguments)]
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

/// Create a ConvTranspose2dConfig from the attributes of the node
pub fn conv_transpose2d_config(curr: &Node) -> ConvTranspose2dConfig {
    let mut attrs = curr.attrs.clone();
    let kernel_shape = attrs
        .remove("kernel_shape")
        .map(AttributeValue::into_i64s)
        .unwrap_or_default();
    let stride = attrs
        .remove("strides")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1]);
    let pads = attrs
        .remove("pads")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0, 0, 0]);
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1, 1]);
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1) as usize;
    let output_padding = attrs
        .remove("output_padding")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0]);

    // Trick with remove + empty check is simplest way to not forget some attribute for runtime:
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
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
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        output_padding: Vec<i64>,
        group: i64,
        has_bias: bool,
    ) -> Node {
        let weight_tensor = TensorData {
            data: Data::Float32s(vec![0.0; 16]), // Not important for the test
            shape: vec![2, 4, 2, 2], // [input_channels, output_channels/groups, k_h, k_w]
        };

        let mut inputs = vec![
            Argument {
                name: "data".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "weight".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: Some(weight_tensor),
                passed: true,
            },
        ];

        if has_bias {
            inputs.push(Argument {
                name: "bias".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            });
        }

        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            AttributeValue::Int64s(kernel_shape),
        );
        attrs.insert("strides".to_string(), AttributeValue::Int64s(strides));
        attrs.insert("pads".to_string(), AttributeValue::Int64s(pads));
        attrs.insert("dilations".to_string(), AttributeValue::Int64s(dilations));
        attrs.insert(
            "output_padding".to_string(),
            AttributeValue::Int64s(output_padding),
        );
        attrs.insert("group".to_string(), AttributeValue::Int64(group));

        Node {
            node_type: NodeType::ConvTranspose2d,
            name: "test_convtranspose2d".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
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
