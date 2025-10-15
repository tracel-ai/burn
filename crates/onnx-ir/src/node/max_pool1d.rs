use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{
    ir::{Node, NodeConfig},
    node::padding::padding_config_1d,
};
use std::any::Any;

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

impl NodeConfig for MaxPool1dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct MaxPool1dProcessor;

impl NodeProcessor for MaxPool1dProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        const MIN: usize = 11;

        // MaxPool implementation supports opset 11+ (for enhanced calculations)
        if opset < MIN {
            return Err(ProcessError::UnsupportedOpset {
                required: MIN,
                actual: opset,
            });
        }

        // Validate input/output count
        if node.inputs.is_empty() {
            return Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: node.inputs.len(),
            });
        }

        if node.outputs.is_empty() {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        let mut kernel_shape = Vec::new();
        let mut stride = vec![1];
        let mut pads = vec![0, 0];
        let mut dilation = vec![1];

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => stride = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilation = value.clone().into_i64s(),
                "auto_pad" => {
                    let auto_pad = value.clone().into_string();
                    if auto_pad != "NOTSET" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "auto_pad".to_string(),
                            reason: format!("Unsupported 'auto_pad' value: {auto_pad}"),
                        });
                    }
                }
                "ceil_mode" => {
                    if value.clone().into_i64() == 1 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "ceil_mode".to_string(),
                            reason: "ceil_mode is not supported".to_string(),
                        });
                    }
                }
                // These are attributes that are allowed but not used in this implementation
                "storage_order" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for MaxPool1d: {key}"),
                    });
                }
            }
        }

        if kernel_shape.len() != 1 {
            return Err(ProcessError::Custom(
                "MaxPool1d: kernel shape must have length 1".to_string(),
            ));
        }
        if dilation.len() != 1 {
            return Err(ProcessError::Custom(
                "MaxPool1d: dilation must have length 1".to_string(),
            ));
        }
        if stride.len() != 1 {
            return Err(ProcessError::Custom(
                "MaxPool1d: stride must have length 1".to_string(),
            ));
        }

        let padding = padding_config_1d(&pads);

        let config = MaxPool1dConfig {
            kernel_size: kernel_shape[0] as usize,
            stride: stride[0] as usize,
            dilation: dilation[0] as usize,
            padding,
        };

        node.config = Some(Box::new(config));

        // Output type is same as input
        crate::util::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1];
        let mut pads = vec![0, 0];
        let mut dilation = vec![1];

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => stride = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilation = value.clone().into_i64s(),
                "auto_pad" => {}
                "ceil_mode" => {}
                "storage_order" => {}
                _ => {}
            }
        }

        let padding = padding_config_1d(&pads);

        let config = MaxPool1dConfig {
            kernel_size: kernel_shape[0] as usize,
            stride: stride[0] as usize,
            dilation: dilation[0] as usize,
            padding,
        };

        Ok(Some(Box::new(config)))
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
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], vec![1], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_max_pool1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![2], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 0, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_not_set() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("NOTSET"));
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_not_supported() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("SAME_UPPER"));
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_max_pool1d_config_with_ceil_mode() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 1, None);
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
