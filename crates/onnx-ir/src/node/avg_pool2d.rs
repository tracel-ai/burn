//! # AveragePool (2D)
//!
//! 2D average pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__AveragePool.html>
//!
//! ## Opset Versions
//! - **Opset 7**: Initial AveragePool operator
//! - **Opset 10**: Added dilations attribute support
//! - **Opset 11**: Updated operator and added count_include_pad attribute
//! - **Opset 19**: Added ceil_mode attribute (not supported in this implementation)
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, TensorType};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for AvgPool2d operations
#[derive(Debug, Clone, new)]
pub struct AvgPool2dConfig {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub strides: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Whether to include padding in the average calculation
    pub count_include_pad: bool,
    /// Dilation [height, width] (opset 10+)
    pub dilation: [usize; 2],
}

/// Node representation for AveragePool2d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct AveragePool2dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: AvgPool2dConfig,
}

pub(crate) struct AvgPool2dProcessor;

impl NodeProcessor for AvgPool2dProcessor {
    type Config = AvgPool2dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate that kernel_shape attribute is present (marked as required in spec)
        // Currently extract_config will panic if kernel_shape is missing or has wrong length
        // TODO: Add test for zero or negative kernel_shape values - spec requires positive values
        // TODO: Add test for zero or negative stride values - spec requires positive values
        // TODO: Add test for asymmetric padding edge cases - padding is validated but edge cases may not be tested

        // Validate attributes before extracting config
        let mut ceil_mode: i64 = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" | "count_include_pad" => {}
                "dilations" => {
                    // Dilations support requires opset 10+
                    let dilations = value.clone().into_i64s();
                    if dilations.iter().any(|&d| d != 1) && opset < 10 {
                        return Err(ProcessError::Custom(format!(
                            "AveragePool: dilations requires opset 10+, got opset {}",
                            opset
                        )));
                    }
                }
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                "auto_pad" => {
                    let auto_pad = value.clone().into_string();
                    if auto_pad != "NOTSET" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "auto_pad".to_string(),
                            reason: format!("Unsupported 'auto_pad' value: {}", auto_pad),
                        });
                    }
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for AvgPool2d: {}", key),
                    });
                }
            }
        }

        if ceil_mode == 1 {
            return Err(ProcessError::InvalidAttribute {
                name: "ceil_mode".to_string(),
                reason: "ceil_mode is not supported".to_string(),
            });
        }

        // Extract input tensor type
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // AvgPool2d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut count_include_pad: i64 = 0;
        let mut dilations = vec![1, 1];

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "count_include_pad" => count_include_pad = value.clone().into_i64(),
                "dilations" => dilations = value.clone().into_i64s(),
                _ => {}
            }
        }

        let padding = padding_config_2d(&pads);

        let config = AvgPool2dConfig::new(
            [kernel_shape[0] as usize, kernel_shape[1] as usize],
            [strides[0] as usize, strides[1] as usize],
            padding,
            count_include_pad == 1,
            [dilations[0] as usize, dilations[1] as usize],
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::AveragePool2d(AveragePool2dNode {
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
        count_include_pad: i64,
        ceil_mode: i64,
        dilations: Option<Vec<i64>>,
    ) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::AveragePool2d, "test_avgpool2d")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 4, None)
            .attr_ints("kernel_shape", kernel_shape)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_int("count_include_pad", count_include_pad)
            .attr_int("ceil_mode", ceil_mode);

        if let Some(dilations) = dilations {
            builder = builder.attr_ints("dilations", dilations);
        }

        builder.build()
    }

    #[test]
    fn test_avg_pool2d_config_basic() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![0, 0, 0, 0], 0, 0, None);
        let mut node = node;
        let processor = AvgPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_avg_pool2d_config_with_padding() {
        let node = create_test_node(vec![2, 2], vec![2, 2], vec![1, 1, 1, 1], 0, 0, None);
        let mut node = node;
        let processor = AvgPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.strides, [2, 2]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_avg_pool2d_config_with_count_include_pad() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![1, 1, 1, 1], 1, 0, None);
        let mut node = node;
        let processor = AvgPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert!(config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_avg_pool2d_config_with_dilation() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            0,
            0,
            Some(vec![2, 2]),
        );
        let mut node = node;
        let processor = AvgPool2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert_eq!(config.strides, [1, 1]);
        assert_eq!(config.dilation, [2, 2]);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_avg_pool2d_config_with_ceil_mode() {
        let node = create_test_node(vec![3, 3], vec![1, 1], vec![0, 0, 0, 0], 0, 1, None);
        let mut node = node;
        let processor = AvgPool2dProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_avg_pool2d_dilation_opset_validation() {
        // Test that opset < 11 is rejected entirely (due to count_include_pad requirement)
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![0, 0, 0, 0],
            0,
            0,
            Some(vec![2, 2]),
        );
        let processor = AvgPool2dProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 10, &spec);
        // Should fail because minimum opset is 11
        assert!(matches!(result, Err(ProcessError::UnsupportedOpset { .. })));
    }
}
