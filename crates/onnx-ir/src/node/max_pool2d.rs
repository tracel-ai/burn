use crate::ir::Node;
use crate::node::padding::{PaddingConfig2d, padding_config_2d};

/// Configuration for MaxPool2d operations
#[derive(Debug, Clone)]
pub struct MaxPool2dConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub strides: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width]
    pub dilation: [usize; 2],
}

impl MaxPool2dConfig {
    /// Create a new MaxPool2dConfig
    pub fn new(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            strides: [1, 1],
            padding: PaddingConfig2d::Valid,
            dilation: [1, 1],
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

    /// Set the dilation
    pub fn with_dilation(mut self, dilation: [usize; 2]) -> Self {
        self.dilation = dilation;
        self
    }
}

/// Create a MaxPool2dConfig from the attributes of the node
pub fn max_pool2d_config(curr: &Node) -> MaxPool2dConfig {
    let mut kernel_shape = Vec::new();
    let mut strides = vec![1, 1];
    let mut pads = vec![0, 0, 0, 0];
    let mut dilations = vec![1, 1];

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "kernel_shape" => kernel_shape = value.clone().into_i64s(),
            "strides" => strides = value.clone().into_i64s(),
            "pads" => pads = value.clone().into_i64s(),
            "dilations" => dilations = value.clone().into_i64s(),
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
            _ => panic!("Unexpected attribute for MaxPool2d: {key}"),
        }
    }

    let padding = padding_config_2d(&pads);

    MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
        .with_strides([strides[0] as usize, strides[1] as usize])
        .with_padding(padding)
        .with_dilation([dilations[0] as usize, dilations[1] as usize])
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
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::MaxPool2d, "test_maxpool2d")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_int("ceil_mode", ceil_mode)
            .attr_ints("dilations", dilations);
        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }
        builder.build()
    }

    #[test]
    fn test_max_pool2d_config_basic() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            0,
            None,
        );
        let config = max_pool2d_config(&node);

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_max_pool2d_config_with_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![2, 2],
            vec![1, 1, 1, 1],
            vec![1, 1],
            0,
            None,
        );
        let config = max_pool2d_config(&node);

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.strides, [2, 2]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_max_pool2d_config_with_dilation() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![2, 2],
            0,
            None,
        );
        let config = max_pool2d_config(&node);

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [2, 2]);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_max_pool2d_config_auto_pad_not_set() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            0,
            Some("NOTSET"),
        );
        let config = max_pool2d_config(&node);

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    #[should_panic = "Unsupported 'auto_pad' value"]
    fn test_max_pool2d_config_auto_pad_not_supported() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            0,
            Some("SAME_UPPER"),
        );
        let _config = max_pool2d_config(&node);
    }

    #[test]
    #[should_panic(expected = "ceil_mode is not supported")]
    fn test_max_pool2d_config_with_ceil_mode() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            None,
        );
        let _config = max_pool2d_config(&node);
    }
}
