use crate::ir::Node;
use crate::node::padding::{PaddingConfig2d, padding_config_2d};

/// Configuration for AvgPool2d operations
#[derive(Debug, Clone)]
pub struct AvgPool2dConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub strides: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Whether to include padding in the average calculation
    pub count_include_pad: bool,
}

impl AvgPool2dConfig {
    /// Create a new AvgPool2dConfig
    pub fn new(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            strides: [1, 1],
            padding: PaddingConfig2d::Valid,
            count_include_pad: true,
        }
    }

    /// Set the strides
    pub fn with_strides(mut self, strides: [usize; 2]) -> Self {
        self.strides = strides;
        self
    }

    /// Set the padding configuration
    pub fn with_padding(mut self, padding: PaddingConfig2d) -> Self {
        self.padding = padding;
        self
    }

    /// Set whether to include padding in the average calculation
    pub fn with_count_include_pad(mut self, count_include_pad: bool) -> Self {
        self.count_include_pad = count_include_pad;
        self
    }
}

/// Create a AvgPool2dConfig from the attributes of the node
pub fn avg_pool2d_config(curr: &Node) -> AvgPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
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

    if ceil_mode == 1 {
        panic!("ceil_mode is not supported");
    }

    let padding = padding_config_2d(&pads);

    AvgPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_count_include_pad(count_include_pad == 1)
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
                rank: 4,
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
            node_type: NodeType::AveragePool2d,
            name: "test_avgpool2d".to_string(),
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
    fn test_avg_pool2d_config_basic() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![0, 0, 0, 0], 0, 0);
        let config = avg_pool2d_config(&node);

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_avg_pool2d_config_with_padding() {
        let node = create_test_node(vec![2, 2], vec![2, 2], vec![1, 1, 1, 1], 0, 0);
        let config = avg_pool2d_config(&node);

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.strides, [2, 2]);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_avg_pool2d_config_with_count_include_pad() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![1, 1, 1, 1], 1, 0);
        let config = avg_pool2d_config(&node);

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert!(config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    #[should_panic(expected = "ceil_mode is not supported")]
    fn test_avg_pool2d_config_with_ceil_mode() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![0, 0, 0, 0], 0, 1);
        let _ = avg_pool2d_config(&node);
    }
}
