use crate::ir::Node;

// Reuse PaddingConfig1d from conv1d module
pub use super::conv1d::PaddingConfig1d;

/// Configuration for MaxPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone)]
pub struct MaxPool1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation
    pub dilation: usize,
    /// Padding configuration
    pub padding: PaddingConfig1d,
}

/// Create a MaxPool1dConfig from the attributes of the node
pub fn max_pool1d_config(curr: &Node) -> MaxPool1dConfig {
    let mut kernel_shape = Vec::new();
    let mut stride = vec![1];
    let mut pads = vec![0, 0];
    let mut dilation = vec![1];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => stride = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilation = value.clone().into_i64s(),
            _ => {}
        }
    }

    assert_eq!(
        kernel_shape.len(),
        1,
        "MaxPool1d: kernel shape must have length 1"
    );
    assert_eq!(dilation.len(), 1, "MaxPool1d: dilation must have length 1");
    assert_eq!(stride.len(), 1, "MaxPool1d: stride must have length 1");

    let padding = super::conv1d::padding_config_1d(&pads);

    MaxPool1dConfig {
        kernel_size: kernel_shape[0] as usize,
        stride: stride[0] as usize,
        dilation: dilation[0] as usize,
        padding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, ElementType, NodeType, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilation: Vec<i64>,
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
        attrs.insert("strides".to_string(), AttributeValue::Int64s(stride));
        attrs.insert("pads".to_string(), AttributeValue::Int64s(pads));
        attrs.insert("dilations".to_string(), AttributeValue::Int64s(dilation));

        Node {
            node_type: NodeType::MaxPool1d,
            name: "test_maxpool1d".to_string(),
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
    fn test_max_pool1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1]);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1]);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_max_pool1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2]);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_max_pool1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1]);
        let _ = max_pool1d_config(&node);
    }
}
