use crate::ir::Node;

use super::padding::{PaddingConfig1d, padding_config_1d};

/// Configuration for Conv1d operations extracted from ONNX nodes
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
    pub fn new(channels_in: usize, channels_out: usize, kernel_size: usize) -> Self {
        Self {
            channels_in,
            channels_out,
            kernel_size,
            stride: 1,
            padding: PaddingConfig1d::Valid,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }

    /// Set the stride
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding configuration
    pub fn with_padding(mut self, padding: PaddingConfig1d) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of groups
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Set whether bias is used
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// Create a Conv1dConfig from the attributes of the node
pub fn conv1d_config(curr: &Node) -> Conv1dConfig {
    let mut kernel_shape = Vec::new(); // TODO default inferred from weight tensor per spec
    let mut strides = vec![1];
    let mut pads = vec![0, 0];
    let mut dilations = vec![1];
    let mut group: usize = 1;

    let weight_shape = curr.inputs[1]
        .value
        .as_ref()
        .expect("Conv1d: weight tensor must be present")
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
            _ => {}
        }
    }

    // the channels are inverted in the weight tensor
    let channels_in = weight_shape[1] * group;
    let channels_out = weight_shape[0];

    let padding = padding_config_1d(&pads);

    Conv1dConfig {
        channels_in,
        channels_out,
        kernel_size: kernel_shape[0] as usize,
        stride: strides[0] as usize,
        dilation: dilations[0] as usize,
        groups: group,
        bias,
        padding,
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
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
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
        attrs.insert("strides".to_string(), AttributeValue::Int64s(strides));
        attrs.insert("pads".to_string(), AttributeValue::Int64s(pads));
        attrs.insert("dilations".to_string(), AttributeValue::Int64s(dilations));
        attrs.insert("group".to_string(), AttributeValue::Int64(group));

        Node {
            node_type: NodeType::Conv1d,
            name: "test_conv1d".to_string(),
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
    fn test_conv1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, false);
        let config = conv1d_config(&node);

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
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1], 1, true);
        let config = conv1d_config(&node);

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
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2], 1, false);
        let config = conv1d_config(&node);

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
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 2, false);
        let config = conv1d_config(&node);

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
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 1, false);
        let _ = conv1d_config(&node);
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_conv1d_config_negative_padding() {
        let node = create_test_node(vec![4], vec![1], vec![-1, -1], vec![1], 1, false);
        let _ = conv1d_config(&node);
    }
}
