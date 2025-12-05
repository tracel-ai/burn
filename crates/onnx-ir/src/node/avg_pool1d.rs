//! # AveragePool (1D)
//!
//! 1D average pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__AveragePool.html>
//!
//! ## Opset Versions
//! - **Opset 7**: Initial AveragePool operator
//! - **Opset 10**: Added dilations attribute support
//! - **Opset 11**: Updated operator and added count_include_pad attribute
//! - **Opset 19**: Added ceil_mode attribute
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, TensorType};
use crate::node::padding::padding_config_1d;
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use super::padding::PaddingConfig1d;

/// Configuration for AvgPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone, new)]
pub struct AvgPool1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding configuration
    pub padding: PaddingConfig1d,
    /// Whether to include padding in the average calculation
    pub count_include_pad: bool,
    /// Dilation (opset 10+)
    pub dilation: usize,
    /// Whether to use ceil mode for output size calculation (opset 19+)
    pub ceil_mode: bool,
}

/// Node representation for AveragePool1d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct AveragePool1dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: AvgPool1dConfig,
}

pub(crate) struct AvgPool1dProcessor;

impl NodeProcessor for AvgPool1dProcessor {
    type Config = AvgPool1dConfig;

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
        // Currently extract_config will panic if kernel_shape is missing
        // TODO: Add test coverage for kernel_shape with wrong length (e.g., [3, 3] for 1D pool)
        // TODO: Add test for zero or negative kernel_shape values - spec requires positive values
        // TODO: Add test for zero or negative stride values - spec requires positive values

        // Validate attributes before extracting config
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" | "count_include_pad" => {}
                "ceil_mode" => {
                    // ceil_mode support requires opset 19+
                    let ceil_mode = value.clone().into_i64();
                    if ceil_mode != 0 && opset < 19 {
                        return Err(ProcessError::Custom(format!(
                            "AveragePool: ceil_mode requires opset 19+, got opset {}",
                            opset
                        )));
                    }
                }
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
                        reason: format!("Unexpected attribute for AvgPool1d: {}", key),
                    });
                }
            }
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

        // AvgPool1d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1];
        let mut pads = vec![0, 0];
        let mut count_include_pad: i64 = 0;
        let mut dilations = vec![1];
        let mut ceil_mode: i64 = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "count_include_pad" => count_include_pad = value.clone().into_i64(),
                "dilations" => dilations = value.clone().into_i64s(),
                "ceil_mode" => ceil_mode = value.clone().into_i64(),
                _ => {}
            }
        }

        let padding = padding_config_1d(&pads);

        let config = AvgPool1dConfig::new(
            kernel_shape[0] as usize,
            strides[0] as usize,
            padding,
            count_include_pad == 1,
            dilations[0] as usize,
            ceil_mode == 1,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::AveragePool1d(AveragePool1dNode {
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
        let mut builder = TestNodeBuilder::new(NodeType::AveragePool1d, "test_avgpool1d")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
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
    fn test_avg_pool1d_config_basic() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(!config.count_include_pad);
        assert!(!config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_avg_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], 0, 0, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.dilation, 1);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_avg_pool1d_config_with_count_include_pad() {
        let node = create_test_node(vec![4], vec![1], vec![2, 2], 1, 0, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Explicit(2)));
    }

    #[test]
    fn test_avg_pool1d_config_with_dilation() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0, Some(vec![2]));
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 2);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_avg_pool1d_config_with_ceil_mode() {
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 1, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        // ceil_mode requires opset 19+
        let config = processor.extract_config(&node, 19).unwrap();
        processor.infer_types(&mut node, 19, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(!config.count_include_pad);
        assert!(config.ceil_mode);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_avg_pool1d_dilation_opset_validation() {
        // Test that opset < 11 is rejected entirely (due to count_include_pad requirement)
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0, Some(vec![2]));
        let processor = AvgPool1dProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 10, &spec);
        // Should fail because minimum opset is 11
        assert!(matches!(result, Err(ProcessError::UnsupportedOpset { .. })));
    }

    #[test]
    fn test_avg_pool1d_ceil_mode_opset_validation() {
        // Test that ceil_mode=1 with opset < 19 is rejected
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 1, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 18, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
        if let Err(ProcessError::Custom(msg)) = result {
            assert!(msg.contains("ceil_mode requires opset 19+"));
        }
    }

    #[test]
    fn test_avg_pool1d_ceil_mode_zero_accepted_old_opset() {
        // Test that ceil_mode=0 is accepted even with old opset
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 11, &prefs);
        assert!(result.is_ok());
    }
}
