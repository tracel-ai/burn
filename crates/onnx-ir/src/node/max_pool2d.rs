//! # MaxPool (2D)
//!
//! 2D max pooling operation.
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
//! **Implementation Note**: This implementation validates opset 11+ (see FIXME at lines 93-94).
//! Only validates 1 output (not the optional Indices output, see FIXME at lines 99-100).
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
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for MaxPool2d operations
#[derive(Debug, Clone, new)]
pub struct MaxPool2dConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub strides: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width]
    pub dilation: [usize; 2],
    /// Whether to use ceil mode for output size calculation (opset 10+)
    pub ceil_mode: bool,
}

/// Node representation for MaxPool2d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MaxPool2dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MaxPool2dConfig,
}

impl MaxPool2dConfig {
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

pub(crate) struct MaxPool2dProcessor;

impl NodeProcessor for MaxPool2dProcessor {
    type Config = MaxPool2dConfig;

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
        // TODO: Validate input tensor is 4D (N x C x H x W) - Lower or higher rank should be rejected - burn/crates/onnx-ir/src/node/max_pool2d.rs:101
        // TODO: Validate input dtype - int8/uint8 support requires opset 12+ - burn/crates/onnx-ir/src/node/max_pool2d.rs:101

        // FIXME: Spec mentions optional second output "Indices" but we only validate 1 output.
        // Should validate that output count is 1 or 2, not exactly 1.

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
                    // ceil_mode support requires opset 10+
                    let ceil_mode = value.clone().into_i64();
                    if ceil_mode != 0 && opset < 10 {
                        return Err(ProcessError::Custom(format!(
                            "MaxPool: ceil_mode requires opset 10+, got opset {}",
                            opset
                        )));
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

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut ceil_mode: i64 = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                "auto_pad" => {}
                "storage_order" => {}
                _ => {}
            }
        }

        let padding = padding_config_2d(&pads);

        let config = MaxPool2dConfig::new(
            [kernel_shape[0] as usize, kernel_shape[1] as usize],
            [strides[0] as usize, strides[1] as usize],
            padding,
            [dilations[0] as usize, dilations[1] as usize],
            ceil_mode == 1,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::MaxPool2d(MaxPool2dNode {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        ceil_mode: i64,
        auto_pad: Option<&str>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::MaxPool2d, "test_maxpool2d")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(!config.ceil_mode);
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        processor.infer_types(&mut node, 16, &prefs).unwrap();

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
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_max_pool2d_ceil_mode_opset_validation() {
        // Test that ceil_mode=1 with opset < 10 is rejected
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
        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
        if let Err(ProcessError::Custom(msg)) = result {
            assert!(msg.contains("ceil_mode requires opset 10+"));
        }
    }

    #[test]
    fn test_max_pool2d_ceil_mode_zero_accepted_old_opset() {
        // Test that ceil_mode=0 is accepted even with old opset
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
        let result = processor.infer_types(&mut node, 1, &prefs);
        assert!(result.is_ok());
    }
}
