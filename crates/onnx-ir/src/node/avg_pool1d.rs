//! # AveragePool (1D)
//!
//! 1D average pooling operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__AveragePool.html>
//!
//! ## Attributes
//! - `kernel_shape` (ints, required): Kernel size
//! - `strides` (ints, default=1): Stride
//! - `pads` (ints, default=0): Padding
//! - `dilations` (ints, default=1): Dilation (opset 10+)
//! - `auto_pad` (string, default="NOTSET"): Padding mode (only `NOTSET` supported)
//! - `count_include_pad` (int, default=0): Include padding in average calculation
//! - `ceil_mode` (int, default=0): Use ceil for output shape (not supported)
//!
//! ## Inputs
//! - `X` (T): Input tensor (N x C x L)
//!
//! ## Outputs
//! - `Y` (T): Output tensor
//!
//! ## Opset Versions
//! - **Opset 7**: Initial AveragePool operator
//! - **Opset 10**: Added dilations attribute support
//! - **Opset 11**: Updated operator and added count_include_pad attribute
//! - **Opset 19**: Added ceil_mode attribute (not supported in this implementation)

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::node::padding::padding_config_1d;
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

use super::padding::PaddingConfig1d;

/// Configuration for AvgPool1d operations extracted from ONNX nodes
#[derive(Debug, Clone)]
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
}

impl AvgPool1dConfig {
    /// Create a new AvgPool1dConfig
    pub fn new(
        kernel_size: usize,
        stride: usize,
        padding: PaddingConfig1d,
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
            dilation: 1,
        }
    }

    /// Set the dilation
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }
}

impl NodeConfig for AvgPool1dConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct AvgPool1dProcessor;

impl NodeProcessor for AvgPool1dProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 11)?;

        // Validate input count
        crate::processor::validate_input_count(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Validate that kernel_shape attribute is present (marked as required in spec)
        // Currently extract_config will panic if kernel_shape is missing
        // TODO: Add test coverage for kernel_shape with wrong length (e.g., [3, 3] for 1D pool)
        // TODO: Add test for zero or negative kernel_shape values - spec requires positive values
        // TODO: Add test for zero or negative stride values - spec requires positive values

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
                        reason: format!("Unexpected attribute for AvgPool1d: {}", key),
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

        // AvgPool1d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1];
        let mut pads = vec![0, 0];
        let mut count_include_pad: i64 = 0;
        let mut dilations = vec![1];

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

        let padding = padding_config_1d(&pads);

        let config = AvgPool1dConfig {
            kernel_size: kernel_shape[0] as usize,
            stride: strides[0] as usize,
            padding,
            count_include_pad: count_include_pad == 1,
            dilation: dilations[0] as usize,
        };

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
        count_include_pad: i64,
        ceil_mode: i64,
        dilations: Option<Vec<i64>>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::AveragePool1d, "test_avgpool1d")
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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<AvgPool1dConfig>();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.dilation, 1);
        assert!(!config.count_include_pad);
        assert!(matches!(config.padding, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_avg_pool1d_config_with_padding() {
        let node = create_test_node(vec![4], vec![2], vec![2, 2], 0, 0, None);
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<AvgPool1dConfig>();

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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<AvgPool1dConfig>();

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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<AvgPool1dConfig>();

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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_avg_pool1d_dilation_opset_validation() {
        // Test that opset < 11 is rejected entirely (due to count_include_pad requirement)
        let node = create_test_node(vec![4], vec![1], vec![0, 0], 0, 0, Some(vec![2]));
        let mut node = node;
        let processor = AvgPool1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 10).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 10, &prefs);
        // Should fail because minimum opset is 11
        assert!(matches!(result, Err(ProcessError::UnsupportedOpset { .. })));
    }
}
