use crate::ir::{AttributeValue, Node};

/// Configuration for ConvTranspose1d operations extracted from ONNX nodes
#[derive(Debug, Clone)]
pub struct ConvTranspose1dConfig {
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
    /// Padding size
    pub padding: usize,
    /// Output padding size
    pub padding_out: usize,
}

/// Create a ConvTranspose1dConfig from the attributes of the node
pub fn conv_transpose1d_config(curr: &Node) -> ConvTranspose1dConfig {
    let mut attrs = curr.attrs.clone();

    // Extract kernel_shape, default to an empty vector if not present
    let kernel_shape = attrs
        .remove("kernel_shape")
        .map(AttributeValue::into_i64s)
        .unwrap_or_default();

    // Extract strides, default to 1 if not present
    let stride = attrs
        .remove("strides")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1]);

    // Extract padding, default to 0 if not present
    let pads = attrs
        .remove("pads")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0, 0]);

    // Extract dilations, default to 1 if not present
    let dilations = attrs
        .remove("dilations")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![1]);

    // Extract group attribute, default to 1
    let group = attrs
        .remove("group")
        .map(AttributeValue::into_i64)
        .unwrap_or(1) as usize;

    // Extract output_padding, default to 0 if not present
    let output_padding = attrs
        .remove("output_padding")
        .map(AttributeValue::into_i64s)
        .unwrap_or_else(|| vec![0]);

    // Ensure no unused attributes remain
    if !attrs.is_empty() {
        panic!("Not all attributes are used: {attrs:?}");
    }

    // Check the pads are symmetric
    if pads.len() != 2 || pads[0] != pads[1] {
        panic!(
            "Asymmetric padding is not supported for ConvTranspose1d: {:?}",
            pads
        );
    }

    let weight_shape = curr.inputs[1]
        .value
        .as_ref()
        .expect("ConvTranspose1d: weight tensor must be present")
        .shape
        .clone();

    // Check if bias is present (third input)
    let bias = curr.inputs.len() == 3;

    // Extract channels from the weight tensor shape [out_channels, in_channels]
    let channels_in = weight_shape[1] * group;
    let channels_out = weight_shape[0];

    ConvTranspose1dConfig {
        channels_in,
        channels_out,
        kernel_size: kernel_shape[0] as usize,
        stride: stride[0] as usize,
        padding: pads[0] as usize,
        dilation: dilations[0] as usize,
        padding_out: output_padding[0] as usize,
        groups: group,
        bias,
    }
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
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        output_padding: Vec<i64>,
        has_bias: bool,
    ) -> Node {
        let mut inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        // Add weight tensor
        inputs.push(Argument {
            name: "weight".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: Some(TensorData {
                data: Data::Float32s(vec![0.1; 16]),
                shape: vec![2, 2, 4], // [out_channels, in_channels, kernel_size]
            }),
            passed: true,
        });

        // Add bias if needed
        if has_bias {
            inputs.push(Argument {
                name: "bias".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(TensorData {
                    data: Data::Float32s(vec![0.1, 0.2]),
                    shape: vec![2],
                }),
                passed: true,
            });
        }

        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            AttributeValue::Int64s(kernel_shape),
        );
        attrs.insert("strides".to_string(), AttributeValue::Int64s(stride));
        attrs.insert("pads".to_string(), AttributeValue::Int64s(pads));
        attrs.insert("dilations".to_string(), AttributeValue::Int64s(dilations));
        attrs.insert("group".to_string(), AttributeValue::Int64(group));
        attrs.insert(
            "output_padding".to_string(),
            AttributeValue::Int64s(output_padding),
        );

        Node {
            node_type: NodeType::ConvTranspose1d,
            name: "test_conv_transpose1d".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_conv_transpose1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, vec![0], false);
        let config = conv_transpose1d_config(&node);

        assert_eq!(config.channels_in, 2);
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 0);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.padding_out, 0);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
    }

    #[test]
    fn test_conv_transpose1d_config_with_params() {
        let node = create_test_node(vec![4], vec![2], vec![1, 1], vec![2], 2, vec![1], true);
        let config = conv_transpose1d_config(&node);

        assert_eq!(config.channels_in, 4); // weight_shape[1] * group = 2 * 2
        assert_eq!(config.channels_out, 2);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.padding, 1);
        assert_eq!(config.dilation, 2);
        assert_eq!(config.padding_out, 1);
        assert_eq!(config.groups, 2);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_conv_transpose1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 1, vec![0], false);
        let _ = conv_transpose1d_config(&node);
    }
}
