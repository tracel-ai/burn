//! # Conv (2D)
//!
//! 2D convolution operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Conv.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic convolution support
//! - **Opset 11**: No changes to Conv operator itself (broader ONNX updates)

use crate::ir::{ArgType, Argument, Node, NodeBuilder, TensorType};
use crate::node::padding::{PaddingConfig2d, padding_config_2d};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Conv2d operation
#[derive(Debug, Clone)]
pub struct Conv2dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: Conv2dConfig,
}

/// Configuration for Conv2d operations
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Channels [in, out]
    pub channels: [usize; 2],
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub stride: [usize; 2],
    /// Padding configuration
    pub padding: PaddingConfig2d,
    /// Dilation [height, width]
    pub dilation: [usize; 2],
    /// Number of groups
    pub groups: usize,
    /// Whether bias is used
    pub bias: bool,
}

impl Conv2dConfig {
    /// Create a new Conv2dConfig
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: PaddingConfig2d,
        dilation: [usize; 2],
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        }
    }
}

/// Node processor for Conv2d operation
pub(crate) struct Conv2dProcessor;

impl NodeProcessor for Conv2dProcessor {
    type Config = Conv2dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(2, 3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Lift weight (input[1]) and optional bias (input[2])
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add test for zero or negative stride values - spec requires positive strides
        // TODO: Add test for zero or negative dilation values - spec requires positive dilations
        // TODO: Add test for zero or negative group values - spec requires positive groups
        // TODO: Validate channels_in divisible by groups - required by spec but not validated
        // TODO: Validate channels_out divisible by groups - required by spec but not validated
        // TODO: Add test for very large kernel_shape/stride/dilation - potential overflow/memory issues
        // TODO: Add test coverage for auto_pad values: SAME_UPPER, SAME_LOWER, VALID - currently unsupported
        // TODO: Add test for asymmetric kernel shapes (e.g., [3, 5]) - valid but may not be tested

        // Validate attributes before extracting config
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" | "strides" | "pads" | "dilations" | "group" => {}
                "auto_pad" => {
                    let auto_pad = value.clone().into_string();
                    if auto_pad != "NOTSET" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "auto_pad".to_string(),
                            reason: format!("Unsupported 'auto_pad' value: {auto_pad}"),
                        });
                    }
                }
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Conv2d: {key}"),
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

        // Validate input tensor rank - Conv2d expects rank 4 (N x C x H x W)
        if tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "Conv2d expects input tensor of rank 4 (N x C x H x W), got rank {}",
                tensor.rank
            )));
        }

        // Validate weight tensor type and rank
        let weight_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor (weight)".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        // Weight should be rank 4 (M x C/group x kH x kW)
        if weight_tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "Conv2d expects weight tensor of rank 4 (M x C/group x kH x kW), got rank {}",
                weight_tensor.rank
            )));
        }

        // Validate dtypes match
        if tensor.dtype != weight_tensor.dtype {
            return Err(ProcessError::TypeMismatch {
                expected: format!("Weight tensor with dtype {:?}", tensor.dtype),
                actual: format!("Weight tensor with dtype {:?}", weight_tensor.dtype),
            });
        }

        // Validate bias if present
        if node.inputs.len() > 2 {
            let bias_tensor = match &node.inputs[2].ty {
                ArgType::Tensor(tensor) => tensor,
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor (bias)".to_string(),
                        actual: format!("{:?}", node.inputs[2].ty),
                    });
                }
            };

            // Bias should be rank 1 (M)
            if bias_tensor.rank != 1 {
                return Err(ProcessError::Custom(format!(
                    "Conv2d expects bias tensor of rank 1 (M), got rank {}",
                    bias_tensor.rank
                )));
            }

            // Validate bias dtype matches
            if tensor.dtype != bias_tensor.dtype {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("Bias tensor with dtype {:?}", tensor.dtype),
                    actual: format!("Bias tensor with dtype {:?}", bias_tensor.dtype),
                });
            }
        }

        // Conv2d preserves rank (same as input)
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut strides = vec![1, 1];
        let mut pads = vec![0, 0, 0, 0];
        let mut dilations = vec![1, 1];
        let mut group: usize = 1;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Conv2d: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        let bias = node.inputs.len() == 3;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => kernel_shape = value.clone().into_i64s(),
                "strides" => strides = value.clone().into_i64s(),
                "pads" => pads = value.clone().into_i64s(),
                "dilations" => dilations = value.clone().into_i64s(),
                "group" => group = value.clone().into_i64() as usize,
                "auto_pad" => {}
                _ => {}
            }
        }

        let channels_in = weight_shape[1] * group;
        let channels_out = weight_shape[0];

        let padding = padding_config_2d(&pads);

        let kernel_size = if kernel_shape.is_empty() {
            if weight_shape.len() != 4 {
                return Err(ProcessError::Custom(format!(
                    "Conv2d: expected to infer kernel shape from a weight tensor of rank 4 but got shape {:?}",
                    weight_shape
                )));
            }
            [weight_shape[2], weight_shape[3]]
        } else {
            [kernel_shape[0] as _, kernel_shape[1] as _]
        };

        let config = Conv2dConfig::new(
            [channels_in, channels_out],
            kernel_size,
            [strides[0] as usize, strides[1] as usize],
            padding,
            [dilations[0] as usize, dilations[1] as usize],
            group,
            bias,
        );

        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Conv2d(Conv2dNode {
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
        group: i64,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> TestNodeBuilder {
        // Weight tensor data - not important for the test
        // [output_channels, input_channels/groups, k_h, k_w]
        let weight_shape = vec![4, 2, 2, 2];
        let weight_data = vec![0.0; 32]; // 4*2*2*2 = 32

        let has_kernel_shape = !kernel_shape.is_empty();

        let mut builder = TestNodeBuilder::new(NodeType::Conv2d, "test_conv2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data.clone(), weight_shape)
            .output_tensor_f32("output", 4, None)
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group);

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        builder
    }

    #[test]
    fn test_conv2d_config_basic() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.channels, [2, 4]);
        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.groups, 1);
        assert!(!config.bias);
        assert!(matches!(config.padding, PaddingConfig2d::Valid));
    }

    #[test]
    fn test_conv2d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_conv2d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.groups, 2);
        assert_eq!(config.channels, [4, 4]); // channels_in is adjusted by groups
    }

    #[test]
    fn test_conv2d_config_with_bias() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            true,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(config.bias);
    }

    #[test]
    fn test_conv2d_config_autopad_not_set() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [3, 3]);
        assert!(matches!(config.padding, PaddingConfig2d::Explicit(1, 1)));
    }

    #[test]
    fn test_conv2d_config_autopad_not_supported() {
        let node = create_test_node(
            vec![3, 3],
            vec![1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_conv2d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Conv2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]); // Inferred via weight tensor shape
    }
}
