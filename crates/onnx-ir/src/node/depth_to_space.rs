//! # DepthToSpace
//!
//! Rearranges (permutes) data from depth into blocks of spatial data. This operation
//! moves values from the depth/channel dimension into spatial blocks in the height and
//! width dimensions. It is the reverse transformation of SpaceToDepth.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__DepthToSpace.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with DCR mode only
//! - **Opset 11**: Added CRD mode support and additional type support
//! - **Opset 13**: Extended type support (bfloat16)
//!
//! ## Implementation Notes
//! - Current implementation validates opset 11+ (see FIXME at line 83)
//! - According to spec, operator exists since opset 1

use onnx_ir_derive::NodeBuilder;

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{ArgType, Argument, Node, NodeBuilder, TensorType};

/// Node representation for DepthToSpace operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct DepthToSpaceNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DepthToSpaceConfig,
}

/// Mode for DepthToSpace operation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DepthToSpaceMode {
    #[default]
    Dcr,
    Crd,
}

impl DepthToSpaceMode {
    fn from_str(val: &str) -> Result<Self, String> {
        match val {
            "DCR" => Ok(Self::Dcr),
            "CRD" => Ok(Self::Crd),
            _ => Err(format!("Unexpected value for DepthToSpace mode: {}", val)),
        }
    }
}

/// Configuration for DepthToSpace operation
#[derive(Debug, Clone)]
pub struct DepthToSpaceConfig {
    pub mode: DepthToSpaceMode,
    pub block_size: usize,
}

impl DepthToSpaceConfig {
    /// Create a new DepthToSpaceConfig
    pub fn new(mode: DepthToSpaceMode, block_size: usize) -> Self {
        Self { mode, block_size }
    }
}

pub(crate) struct DepthToSpaceProcessor;

impl NodeProcessor for DepthToSpaceProcessor {
    type Config = DepthToSpaceConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate unexpected attributes before config extraction
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "blocksize" | "mode" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for DepthToSpace: {}", key),
                    });
                }
            }
        }

        // Get reference to config for type inference
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");

        // Validate that if mode is CRD, we need opset 11+
        if config.mode == DepthToSpaceMode::Crd && opset < 11 {
            return Err(ProcessError::Custom(format!(
                "DepthToSpace: CRD mode requires opset 11+, got opset {}",
                opset
            )));
        }
        let block_size = config.block_size;

        // Validate block_size
        if block_size == 0 {
            return Err(ProcessError::InvalidAttribute {
                name: "blocksize".to_string(),
                reason: "block_size must be greater than 0".to_string(),
            });
        }
        // TODO: Validate that C (channels) is divisible by (blocksize * blocksize) - Per ONNX spec, C must be divisible by blocksize^2 or result is undefined - Should validate in static shape case and add test for invalid channel count

        // Extract the input tensor type to determine rank and shape
        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        if tensor.rank != 4 {
            return Err(ProcessError::Custom(
                "DepthToSpace: only rank 4 tensors are supported".to_string(),
            ));
        }

        // Infer static shape based on rank and block size
        let static_shape = tensor.static_shape.clone().map(|shape| {
            let [b, c, h, w] = shape
                .try_into()
                .expect("DepthToSpace: input tensor rank is not 4");
            vec![
                b,
                c / (block_size * block_size),
                h * block_size,
                w * block_size,
            ]
        });

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        let mut block_size: Option<usize> = None;
        let mut mode = DepthToSpaceMode::Dcr;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "blocksize" => block_size = Some(value.clone().into_i64() as usize),
                "mode" => {
                    mode =
                        DepthToSpaceMode::from_str(&value.clone().into_string()).map_err(|e| {
                            ProcessError::InvalidAttribute {
                                name: "mode".to_string(),
                                reason: e,
                            }
                        })?;
                }
                _ => {}
            }
        }

        let block_size =
            block_size.ok_or_else(|| ProcessError::MissingAttribute("blocksize".to_string()))?;

        let config = DepthToSpaceConfig::new(mode, block_size);
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::DepthToSpace(DepthToSpaceNode {
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
    use crate::ir::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(
        rank: usize,
        static_shape: Option<Vec<usize>>,
        block_size: i64,
        mode: Option<&str>,
    ) -> NodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::DepthToSpace, "test_depth_to_space")
            .input_tensor_f32("input", rank, static_shape)
            .output_tensor_f32("output", rank, None) // Same rank as input
            .attr_int("blocksize", block_size);

        // Add mode attribute if provided
        if let Some(mode_str) = mode {
            builder = builder.attr_string("mode", mode_str);
        }

        builder.build()
    }

    #[test]
    fn test_basic_config() {
        let node = create_test_node(4, None, 2, None);
        let mut node = node;
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.block_size, 2);
        assert_eq!(config.mode, DepthToSpaceMode::Dcr);
    }

    #[test]
    fn test_dcr_config() {
        let node = create_test_node(4, None, 3, Some("DCR"));
        let mut node = node;
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::Dcr);
    }

    #[test]
    fn test_crd_config() {
        let node = create_test_node(4, None, 3, Some("CRD"));
        let mut node = node;
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::Crd);
    }

    #[test]
    fn test_static_shape_update_outputs() {
        let mut node = create_test_node(4, Some(vec![2, 4, 2, 3]), 2, None);
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, vec![2, 1, 4, 6].into());
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    // TODO: Add test for invalid channel count - Test case where C is not divisible by blocksize^2 (e.g., C=5, blocksize=2) should return error - Missing test coverage for constraint validation
    // TODO: Add test for edge case with blocksize=1 - Should be identity operation per spec - Missing edge case test
    // TODO: Add test for larger blocksize values (e.g., 3, 4) - Only testing blocksize=2 and 3, need more coverage - Tests needed for blocksize=4 or higher
    // TODO: Add test for CRD mode with opset < 11 - Should fail per spec, CRD mode added in opset 11 - Missing opset version validation test
    // TODO: Add test for different data types - Spec supports multiple types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float, double, bfloat16) - Only testing f32, need broader type coverage
    // TODO: Add test for zero-size spatial dimensions - Edge case where H=0 or W=0 - Missing edge case test
    // TODO: Add test for very large spatial dimensions - Potential overflow in shape calculation (h * blocksize, w * blocksize) - Missing boundary condition test
}
