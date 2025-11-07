//! # SpaceToDepth
//!
//! Rearranges blocks of spatial data into depth. This operation moves values from the
//! height and width dimensions into the depth/channel dimension. It is the reverse
//! transformation of DepthToSpace.
//!
//! More specifically, this operator outputs a copy of the input tensor where values from
//! the height and width dimensions are moved to the depth dimension. The spatial dimensions
//! are reduced by a factor of `blocksize`, while the depth is increased by `blocksize^2`.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html>
//!
//! ## Opset Versions
//! - **Opset 1-12**: Initial version with blocksize attribute
//! - **Opset 13+**: Extended type support (added bfloat16, uint types)

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::{
    ArgType, TensorType,
    ir::{Node, NodeConfig},
};
use std::any::Any;

/// Configuration for SpaceToDepth operations
#[derive(Debug, Clone)]
pub struct SpaceToDepthConfig {
    /// Block size for space-to-depth transformation
    pub block_size: usize,
}

impl NodeConfig for SpaceToDepthConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct SpaceToDepthProcessor;

impl NodeProcessor for SpaceToDepthProcessor {
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
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate unexpected attributes before config extraction
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "blocksize" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for SpaceToDepth: {}", key),
                    });
                }
            }
        }

        // Get reference to config for type inference
        let config = node.config::<SpaceToDepthConfig>();
        let block_size = config.block_size;

        // Validate block_size
        if block_size == 0 {
            return Err(ProcessError::InvalidAttribute {
                name: "blocksize".to_string(),
                reason: "block_size must be greater than 0".to_string(),
            });
        }

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

        // TODO: Missing validation that input is rank 4 with NCHW format.
        // ONNX spec requires input to be 4D [N, C, H, W] but only rank is checked, not semantics.

        if tensor.rank != 4 {
            return Err(ProcessError::Custom(
                "SpaceToDepth: only rank 4 tensors are supported".to_string(),
            ));
        }

        // TODO: Missing validation that H and W are divisible by blocksize.
        // ONNX spec requires H % blocksize == 0 and W % blocksize == 0, but this isn't validated.
        // Should check when static_shape is available to catch errors early.

        // Infer static shape based on rank and block size
        let static_shape = tensor.static_shape.clone().map(|shape| {
            let [b, c, h, w] = shape
                .try_into()
                .expect("SpaceToDepth: input tensor rank is not 4");
            vec![
                b,
                c * block_size * block_size,
                h / block_size,
                w / block_size,
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
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut block_size: Option<usize> = None;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "blocksize" {
                block_size = Some(value.clone().into_i64() as usize)
            }
        }

        let block_size =
            block_size.ok_or_else(|| ProcessError::MissingAttribute("blocksize".to_string()))?;

        let config = SpaceToDepthConfig { block_size };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(rank: usize, static_shape: Option<Vec<usize>>, block_size: i64) -> Node {
        let builder = NodeBuilder::new(NodeType::DepthToSpace, "test_space_to_depth")
            .input_tensor_f32("input", rank, static_shape)
            .output_tensor_f32("output", rank, None) // Same rank as input
            .attr_int("blocksize", block_size);
        builder.build()
    }

    #[test]
    fn test_basic_config() {
        let node = create_test_node(4, None, 2);
        let mut node = node;
        let processor = SpaceToDepthProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SpaceToDepthConfig>();

        assert_eq!(config.block_size, 2);
    }

    #[test]
    fn test_static_shape_update_outputs() {
        let mut node = create_test_node(4, Some(vec![2, 1, 4, 6]), 2);
        let processor = SpaceToDepthProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, vec![2, 4, 2, 3].into());
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    // TODO: Missing test for blocksize validation - blocksize must be > 0.
    // Currently validated but not explicitly tested.

    // TODO: Missing test for non-divisible dimensions - H or W not divisible by blocksize.
    // E.g., input [1, 1, 5, 6], blocksize=2 should fail (5 % 2 != 0).

    // TODO: Missing test for blocksize=1 edge case - should be no-op transformation.

    // TODO: Missing test for large blocksize - e.g., blocksize > H or blocksize > W.
    // Should be rejected as dimensions would become negative.

    // TODO: Missing test for different data types - verify works with int8, float16, etc.
    // Implementation should support all types per ONNX spec (opset 13+).
}
