//! # MaxPool (1D)
//!
//! 1D max pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MaxPool.html>
//!
//! ## Attributes
//! - `kernel_shape` (ints, required): Kernel size
//! - `strides` (ints, default=1): Stride
//! - `pads` (ints, default=0): Padding
//! - `dilations` (ints, default=1): Dilation
//! - `auto_pad` (string, default="NOTSET"): Padding mode (only `NOTSET` supported)
//! - `ceil_mode` (int, default=0): Use ceil for output shape (not supported)
//! - `storage_order` (int, default=0): Memory layout (only row major supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x L)
//!
//! ## Outputs
//! - `Y` (T): Output tensor
//! - `Indices` (I, optional): Indices tensor
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic max pooling operation.
//! - **Opset 8**: Added support for `storage_order` attribute.
//! - **Opset 10**: Added `ceil_mode` attribute to use ceiling instead of floor for output shape calculation.
//! - **Opset 11**: Added support for dilation; updated padding semantics; added optional Indices output.
//! - **Opset 12**: Added support for int8, uint8 data types; clarified behavior with negative padding.
//!
//! **Implementation Note**: This implementation validates opset 11+ (see FIXME at lines 97-98).
//! The implementation does not support `ceil_mode=1` and only validates 1 output (not the optional
//! Indices output, see FIXME at lines 103-104).

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
        // Spec: Opset 1+ (dilation support added in opset 11)
        crate::processor::validate_opset(opset, 1)?;

        // FIXME: Spec mentions optional second output "Indices" but we only validate 1 output.
        // Should validate that output count is 1 or 2, not exactly 1.

        // Validate input/output count
        crate::processor::validate_min_inputs(node, 1)?;

        crate::processor::validate_output_count(node, 1)?;

        // Validate attributes before extracting config
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" | "storage_order" => {}
                "dilations" => {
                    // Dilation support requires opset 11+
                    let dilations = value.clone().into_i64s();
                    if dilations.iter().any(|&d| d != 1) && opset < 11 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: dilation requires opset 11+, got opset {}",
                            opset
                        )));
                    }
                }
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
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for MaxPool1d: {key}"),
                    });
                }
            }
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    #[should_panic(expected = "Asymmetric padding is not supported")]
    fn test_max_pool1d_config_asymmetric_padding() {
        let node = create_test_node(vec![4], vec![1], vec![1, 2], vec![1], 0, None);
        let processor = MaxPool1dProcessor;
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_max_pool1d_config_auto_pad_not_set() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], vec![1], 0, Some("NOTSET"));
        let mut node = node;
        let processor = MaxPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
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
