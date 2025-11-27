//! # MaxPool (1D)
//!
//! 1D max pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MaxPool.html>
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
//!
//! ## Missing Test Coverage
//! - TODO: No test for dilation > 1 with opset < 11 - Should reject dilation in older opsets
//! - TODO: No test for storage_order != 0 - Non-row-major order should be validated/rejected
//! - TODO: No test for int8/uint8 dtypes - Opset 12+ supports integer types
//! - TODO: No test for kernel_shape validation - Missing kernel_shape attribute should be rejected
//! - TODO: No test for negative padding values - Opset 12+ allows negative padding
//! - TODO: No test for edge case: kernel larger than input dimension
//! - TODO: No test validating input is 3D (N x C x L) - Lower/higher rank should be rejected
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use crate::{
    ir::{Argument, Node, RawNode},
    node::padding::padding_config_1d,
};

use super::padding::PaddingConfig1d;

/// Configuration for MaxPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone, new)]
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

/// Node representation for MaxPool1d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MaxPool1dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MaxPool1dConfig,
}

impl MaxPool1dConfig {
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

pub(crate) struct MaxPool1dProcessor;

impl NodeProcessor for MaxPool1dProcessor {
    type Config = MaxPool1dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input tensor is 3D (N x C x L) - Lower or higher rank should be rejected - burn/crates/onnx-ir/src/node/max_pool1d.rs:105
        // TODO: Validate input dtype - int8/uint8 support requires opset 12+ - burn/crates/onnx-ir/src/node/max_pool1d.rs:105

        // FIXME: Spec mentions optional second output "Indices" but we only validate 1 output.
        // Should validate that output count is 1 or 2, not exactly 1.

        // Validate attributes before extracting config
        // TODO: Validate required kernel_shape attribute is present - Missing kernel_shape should cause error - burn/crates/onnx-ir/src/node/max_pool1d.rs:117

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" => {}
                "storage_order" => {
                    // TODO: Validate storage_order == 0 (row-major) - Non-zero values not supported - burn/crates/onnx-ir/src/node/max_pool1d.rs:119
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
                        reason: format!("Unexpected attribute for MaxPool1d: {key}"),
                    });
                }
            }
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
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

        let config = MaxPool1dConfig::new(
            kernel_shape[0] as usize,
            stride[0] as usize,
            dilation[0] as usize,
            padding,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::MaxPool1d(MaxPool1dNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::NodeType, node::padding::PaddingConfig1d, node::test_utils::TestNodeBuilder};

    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilation: Vec<i64>,
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::MaxPool1d, "test_maxpool1d")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
