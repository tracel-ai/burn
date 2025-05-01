use crate::{ir::Node, node::padding::padding_config_1d};

use super::padding::PaddingConfig1d;

/// Configuration for AvgPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone)]
pub struct AvgPool1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding configuration
    pub padding: PaddingConfig1d,
    /// Whether to include padding in the average calculation
    pub count_include_pad: bool,
}

impl AvgPool1dConfig {
    /// Create a new AvgPool1dConfig
    pub fn new(
        kernel_size: usize,
        stride: usize,
        padding: PaddingConfig1d,
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        }
    }
}

/// Create an AvgPool1dConfig from the attributes of the node
pub fn avg_pool1d_config(curr: &Node) -> AvgPool1dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1];
    let mut pads = vec![0, 0];
    let mut count_include_pad: i64 = 0;
    let mut ceil_mode: i64 = 0;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "count_include_pad" => count_include_pad = value.clone().into_i64(),
            "ceil_mode" => ceil_mode = value.clone().into_i64(),
            _ => {}
        }
    }

    assert_eq!(
        kernel_shape.len(),
        1,
        "AvgPool1d: kernel shape must have length 1"
    );
    assert_eq!(strides.len(), 1, "AvgPool1d: stride must have length 1");

    if ceil_mode == 1 {
        panic!("ceil_mode is not supported");
    }

    let padding = padding_config_1d(&pads);

    AvgPool1dConfig {
        kernel_size: kernel_shape[0] as usize,
        stride: strides[0] as usize,
        padding,
        count_include_pad: count_include_pad == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        count_include_pad: i64,
        ceil_mode: i64,
    ) -> Node {
        let inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            AttributeValue::Int64s(kernel_shape),
        );
        attrs.insert("strides".to_string(), AttributeValue::Int64s(strides));
        attrs.insert("pads".to_string(), AttributeValue::Int64s(pads));
        attrs.insert(
            "count_include_pad".to_string(),
            AttributeValue::Int64(count_include_pad),
        );
        attrs.insert("ceil_mode".to_string(), AttributeValue::Int64(ceil_mode));

        Node {
            node_type: NodeType::AveragePool1d,
            name: "test_avgpool1d".to_string(),
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
    fn test_avg_pool1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0);
        let config = avg_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_avg_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], 0, 0);
        let config = avg_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_avg_pool1d_config_with_count_include_pad() {
        let node = create_test_node(vec![4], vec![1], vec![2, 2], 1, 0);
        let config = avg_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert!(config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    #[should_panic(expected = "ceil_mode is not supported")]
    fn test_avg_pool1d_config_with_ceil_mode() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 1);
        let _ = avg_pool1d_config(&node);
    }
}
