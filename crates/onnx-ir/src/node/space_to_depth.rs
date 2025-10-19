use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

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
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

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

        log::debug!("SpaceToDepth rank inference for node {}", &node.name);

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
                "SpaceToDepth: only rank 4 tensors are supported".to_string(),
            ));
        }

        log::debug!(
            "SpaceToDepth blocksize from attribute for {}: {:?}",
            &node.name,
            block_size
        );

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
            elem_type: tensor.elem_type.clone(),
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
    use crate::ElementType;
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
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
