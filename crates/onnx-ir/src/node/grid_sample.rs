//! # GridSample
//!
//! Performs spatial sampling using normalized grid coordinates.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__GridSample.html>
//!
//! ## Opset Versions
//! - **Opset 16**: Initial version with 4D spatial inputs only (N, C, H, W).
//! - **Opset 20**: Extended to N-dimensional spatial inputs.
//! - **Opset 22**: Added bfloat16 support for grid.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;
use std::str::FromStr;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Interpolation mode for grid sampling
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GridSampleMode {
    /// Bilinear interpolation (default)
    #[default]
    Bilinear,
    /// Nearest neighbor interpolation
    Nearest,
    /// Bicubic interpolation
    Bicubic,
}

impl FromStr for GridSampleMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bilinear" | "linear" => Ok(GridSampleMode::Bilinear),
            "nearest" => Ok(GridSampleMode::Nearest),
            "bicubic" | "cubic" => Ok(GridSampleMode::Bicubic),
            _ => Err(format!("Unsupported grid sample mode: {}", s)),
        }
    }
}

/// Padding mode for out-of-bounds grid values
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GridSamplePaddingMode {
    /// Fill with zeros (default)
    #[default]
    Zeros,
    /// Use border values
    Border,
    /// Reflect at boundaries
    Reflection,
}

impl FromStr for GridSamplePaddingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "zeros" => Ok(GridSamplePaddingMode::Zeros),
            "border" => Ok(GridSamplePaddingMode::Border),
            "reflection" => Ok(GridSamplePaddingMode::Reflection),
            _ => Err(format!("Unsupported grid sample padding mode: {}", s)),
        }
    }
}

/// Configuration for the GridSample operation.
#[derive(Debug, Clone, new)]
pub struct GridSampleConfig {
    /// Interpolation mode (default: bilinear)
    pub mode: GridSampleMode,
    /// Padding mode for out-of-bounds values (default: zeros)
    pub padding_mode: GridSamplePaddingMode,
    /// Whether grid extrema (-1, 1) refer to corner centers (1) or corner points (0)
    /// Default: 0 (corner points)
    pub align_corners: bool,
}

impl Default for GridSampleConfig {
    fn default() -> Self {
        Self {
            mode: GridSampleMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: false,
        }
    }
}

/// Node representation for GridSample operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GridSampleNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: GridSampleConfig,
}

pub(crate) struct GridSampleProcessor;

impl NodeProcessor for GridSampleProcessor {
    type Config = GridSampleConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 16,
            max_opset: None,
            inputs: InputSpec::Exact(2), // X and grid
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get input tensor type
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Get grid tensor type
        let grid_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        // For opset 16, input must be 4D (N, C, H, W)
        if opset < 20 && input_tensor.rank != 4 {
            return Err(ProcessError::Custom(format!(
                "GridSample: For opset {}, input must be 4D (N, C, H, W), got rank {}",
                opset, input_tensor.rank
            )));
        }

        // Grid should have rank = input_rank - 1 + 1 = input_rank
        // For 4D input (N, C, H, W), grid is (N, H_out, W_out, 2)
        let expected_grid_rank = input_tensor.rank;
        if grid_tensor.rank != expected_grid_rank {
            return Err(ProcessError::Custom(format!(
                "GridSample: Grid rank should be {}, got {}",
                expected_grid_rank, grid_tensor.rank
            )));
        }

        // Extract and validate config
        let config = self.extract_config(node, opset)?;

        // Validate bicubic is only for 4D
        if config.mode == GridSampleMode::Bicubic && input_tensor.rank != 4 {
            return Err(ProcessError::Custom(
                "GridSample: Bicubic mode only supports 4D input".to_string(),
            ));
        }

        // Output has same dtype as input, same rank
        // Shape is (N, C, D1_out, D2_out, ...) based on grid dimensions
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: input_tensor.dtype,
            rank: input_tensor.rank,
            static_shape: None, // Output shape depends on grid dimensions
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut mode = GridSampleMode::default();
        let mut padding_mode = GridSamplePaddingMode::default();
        let mut align_corners = false;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "mode" => {
                    mode = value
                        .clone()
                        .into_string()
                        .parse::<GridSampleMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "mode".to_string(),
                            reason: e,
                        })?;
                }
                "padding_mode" => {
                    padding_mode = value
                        .clone()
                        .into_string()
                        .parse::<GridSamplePaddingMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "padding_mode".to_string(),
                            reason: e,
                        })?;
                }
                "align_corners" => {
                    align_corners = value.clone().into_i32() != 0;
                }
                _ => {}
            }
        }

        Ok(GridSampleConfig {
            mode,
            padding_mode,
            align_corners,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::GridSample(GridSampleNode {
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
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::GridSample, "test_grid_sample")
            .input_tensor_f32("X", 4, None) // N, C, H, W
            .input_tensor_f32("grid", 4, None) // N, H_out, W_out, 2
            .output_tensor_f32("Y", 4, None)
    }

    #[test]
    fn test_grid_sample_default_config() {
        let node = create_test_node().build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.mode, GridSampleMode::Bilinear);
        assert_eq!(config.padding_mode, GridSamplePaddingMode::Zeros);
        assert!(!config.align_corners);
    }

    #[test]
    fn test_grid_sample_custom_config() {
        let node = create_test_node()
            .attr_string("mode", "nearest")
            .attr_string("padding_mode", "border")
            .attr_int("align_corners", 1)
            .build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.mode, GridSampleMode::Nearest);
        assert_eq!(config.padding_mode, GridSamplePaddingMode::Border);
        assert!(config.align_corners);
    }

    #[test]
    fn test_grid_sample_bicubic_mode() {
        let node = create_test_node().attr_string("mode", "bicubic").build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.mode, GridSampleMode::Bicubic);
    }

    #[test]
    fn test_grid_sample_type_inference() {
        let mut node = create_test_node().build();
        let processor = GridSampleProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_grid_sample_invalid_input_rank() {
        let mut node = TestNodeBuilder::new(NodeType::GridSample, "test_grid_sample")
            .input_tensor_f32("X", 3, None) // Wrong rank for opset 16
            .input_tensor_f32("grid", 3, None)
            .output_tensor_f32("Y", 3, None)
            .build();
        let processor = GridSampleProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);

        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_grid_sample_linear_mode_alias() {
        // Opset 20+ uses "linear" instead of "bilinear"
        let node = create_test_node().attr_string("mode", "linear").build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 20).unwrap();

        // "linear" should map to Bilinear
        assert_eq!(config.mode, GridSampleMode::Bilinear);
    }

    #[test]
    fn test_grid_sample_cubic_mode_alias() {
        // "cubic" is an alias for "bicubic"
        let node = create_test_node().attr_string("mode", "cubic").build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.mode, GridSampleMode::Bicubic);
    }

    #[test]
    fn test_grid_sample_reflection_padding() {
        let node = create_test_node()
            .attr_string("padding_mode", "reflection")
            .build();
        let processor = GridSampleProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.padding_mode, GridSamplePaddingMode::Reflection);
    }
}
