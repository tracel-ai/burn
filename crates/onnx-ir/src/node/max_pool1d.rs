use crate::{ir::Node, node::padding::padding_config_1d};

use super::padding::PaddingConfig1d;

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

impl MaxPool1dConfig {
    /// Create a new MaxPool1dConfig
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: 1,
            padding: PaddingConfig1d::Valid,
            dilation: 1,
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
            "auto_pad" => {
                let auto_pad = value.clone().into_string();
                if auto_pad != "NOTSET" {
                    panic!("Unsupported 'auto_pad' value: {auto_pad}");
                }
            }
            "ceil_mode" => {
                if value.clone().into_i64() == 1 {
                    panic!("ceil_mode is not supported");
                }
            }
            // These are attributes that are allowed but not used in this implementation
            "storage_order" => {}
            _ => panic!("Unexpected attribute for MaxPool1d: {key}"),
        }
    }

    assert_eq!(
        kernel_shape.len(),
        1,
        "MaxPool1d: kernel shape must have length 1"
    );
    assert_eq!(dilation.len(), 1, "MaxPool1d: dilation must have length 1");
    assert_eq!(stride.len(), 1, "MaxPool1d: stride must have length 1");

    let padding = padding_config_1d(&pads);

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
    use crate::{ir::NodeType, node::padding::PaddingConfig1d, node::test_utils::NodeBuilder};

    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilation: Vec<i64>,
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::MaxPool1d, "test_maxpool1d")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", stride)
            .attr_ints("pads", pads)
            .attr_int("ceil_mode", ceil_mode)
            .attr_ints("dilations", dilation);
        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }
        builder.build()
    }

    #[test]
    fn test_max_pool1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, None);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1], 0, None);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_max_pool1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2], 0, None);
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_max_pool1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 0, None);
        let _ = max_pool1d_config(&node);
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_not_set() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("NOTSET"));
        let config = max_pool1d_config(&node);

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    #[should_panic = "Unsupported 'auto_pad' value"]
    fn test_max_pool1d_config_auto_pad_not_supported() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("SAME_UPPER"));
        let _config = max_pool1d_config(&node);
    }

    #[test]
    #[should_panic(expected = "ceil_mode is not supported")]
    fn test_max_pool1d_config_with_ceil_mode() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, None);
        let _config = max_pool1d_config(&node);
    }
}
