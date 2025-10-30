//! # MaxPool (2D)
//!
//! 2D max pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MaxPool.html>
//!
//! ## Attributes
//! - `kernel_shape` (required, ints): Kernel size \[height, width\]
//! - `strides` (ints, default=\[1, 1\]): Stride \[height, width\]
//! - `pads` (ints, default=\[0, 0, 0, 0\]): Padding \[top, left, bottom, right\]
//! - `dilations` (ints, default=\[1, 1\]): Dilation \[height, width\]
//! - `auto_pad` (string, default="NOTSET"): Padding mode (only `NOTSET` supported)
//! - `ceil_mode` (int, default=0): Use ceil for output shape (not supported)
//! - `storage_order` (int, default=0): Memory layout (only row major supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x H x W)
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
//! **Implementation Note**: This implementation validates opset 11+ (see FIXME at lines 93-94).
//! The implementation does not support `ceil_mode=1` and only validates 1 output (not the optional
//! Indices output, see FIXME at lines 99-100).
//!
//! ## Missing Test Coverage
//! - TODO: No test for dilation > 1 with opset < 11 - Should reject dilation in older opsets
//! - TODO: No test for storage_order != 0 - Non-row-major order should be validated/rejected
//! - TODO: No test for int8/uint8 dtypes - Opset 12+ supports integer types
//! - TODO: No test for kernel_shape validation - Missing kernel_shape attribute should be rejected
//! - TODO: No test for negative padding values - Opset 12+ allows negative padding
//! - TODO: No test for edge case: kernel larger than input dimension
//! - TODO: No test validating input is 4D (N x C x H x W) - Lower/higher rank should be rejected
//! - TODO: No test for asymmetric kernel sizes - e.g., kernel=[3, 5]

use crate::ir::{Node, NodeConfig};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

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

impl NodeConfig for MaxPool2dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct MaxPool2dProcessor;

impl NodeProcessor for MaxPool2dProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Spec: Opset 1+ (dilation support added in opset 11)
        crate::processor::validate_opset(opset, 1)?;

        // TODO: Validate input tensor is 4D (N x C x H x W) - Lower or higher rank should be rejected - burn/crates/onnx-ir/src/node/max_pool2d.rs:101
        // TODO: Validate input dtype - int8/uint8 support requires opset 12+ - burn/crates/onnx-ir/src/node/max_pool2d.rs:101

        // FIXME: Spec mentions optional second output "Indices" but we only validate 1 output.
        // Should validate that output count is 1 or 2, not exactly 1.

        // Validate input/output count
        crate::processor::validate_min_inputs(node, 1)?;

        crate::processor::validate_output_count(node, 1)?;

        // Validate attributes before extracting config
        // TODO: Validate required kernel_shape attribute is present - Missing kernel_shape should cause error - burn/crates/onnx-ir/src/node/max_pool2d.rs:112
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" => {}
                "storage_order" => {
                    // TODO: Validate storage_order == 0 (row-major) - Non-zero values not supported - burn/crates/onnx-ir/src/node/max_pool2d.rs:114
                }
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
                        reason: format!("Unexpected attribute for MaxPool2d: {key}"),
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
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "auto_pad" => {}
                "ceil_mode" => {}
                "storage_order" => {}
                _ => {}
            }
        }

        let padding = padding_config_2d(&pads);

        let config = MaxPool2dConfig::new([kernel_shape[0] as usize, kernel_shape[1] as usize])
            .with_strides([strides[0] as usize, strides[1] as usize])
            .with_padding(padding)
            .with_dilation([dilations[0] as usize, dilations[1] as usize]);

        Ok(Some(Box::new(config)))
    }
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
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool2dConfig>();

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
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool2dConfig>();

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
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool2dConfig>();

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
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<MaxPool2dConfig>();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_max_pool2d_config_auto_pad_not_supported() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            0,
            Some("SAME_UPPER"),
        );
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_max_pool2d_config_with_ceil_mode() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            None,
        );
        let mut node = node;
        let processor = MaxPool2dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }
}
