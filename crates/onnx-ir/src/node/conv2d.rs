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
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
        Self {
            channels,
            kernel_size,
            stride: [1, 1],
            padding: PaddingConfig2d::Valid,
            dilation: [1, 1],
            groups: 1,
            bias: true,
        }
    }

    /// Set the stride
    pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding configuration
    pub fn with_padding(mut self, padding: PaddingConfig2d) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation
    pub fn with_dilation(mut self, dilation: [usize; 2]) -> Self {
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
            _ => {}
        }
    }

    // the channels are inverted in the weight tensor
    let channels_in = weight_shape[1] * group;
    let channels_out = weight_shape[0];

    let padding = padding_config_2d(&pads);

    Conv2dConfig::new(
        [channels_in, channels_out],
        [kernel_shape[0] as usize, kernel_shape[1] as usize],
    )
    .with_stride([strides[0] as usize, strides[1] as usize])
    .with_dilation([dilations[0] as usize, dilations[1] as usize])
    .with_groups(group)
    .with_bias(bias)
    .with_padding(padding)
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
        let weight_tensor = TensorData {
            data: Data::Float32s(vec![0.0; 16]), // Not important for the test
            shape: vec![4, 2, 2, 2], // [output_channels, input_channels/groups, k_h, k_w]
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
        attrs.insert("group".to_string(), AttributeValue::Int64(group));

        Node {
            node_type: NodeType::Conv2d,
            name: "test_conv2d".to_string(),
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
