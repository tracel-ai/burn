use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use crate::{
    ArgType, TensorType,
    ir::{Node, NodeConfig},
};
use std::any::Any;

/// Mode for DepthToSpace operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepthToSpaceMode {
    DCR,
    CRD,
}

impl DepthToSpaceMode {
    fn from_str(val: &str) -> Result<Self, String> {
        match val {
            "DCR" => Ok(Self::DCR),
            "CRD" => Ok(Self::CRD),
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

impl NodeConfig for DepthToSpaceConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct DepthToSpaceProcessor;

impl NodeProcessor for DepthToSpaceProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 11)?;
        crate::util::validate_input_count(node, 1)?;
        crate::util::validate_output_count(node, 1)?;

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
        let config = node.config::<DepthToSpaceConfig>();
        let block_size = config.block_size;

        // Validate block_size
        if block_size == 0 {
            return Err(ProcessError::InvalidAttribute {
                name: "blocksize".to_string(),
                reason: "block_size must be greater than 0".to_string(),
            });
        }

        log::debug!("DepthToSpace rank inference for node {}", &node.name);

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

        log::debug!(
            "DepthToSpace blocksize from attribute for {}: {:?}",
            &node.name,
            block_size
        );

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
        let mut mode = DepthToSpaceMode::DCR;

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
    fn create_test_node(
        rank: usize,
        static_shape: Option<Vec<usize>>,
        block_size: i64,
        mode: Option<&str>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::DepthToSpace, "test_depth_to_space")
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
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DepthToSpaceConfig>();

        assert_eq!(config.block_size, 2);
        assert_eq!(config.mode, DepthToSpaceMode::DCR);
    }

    #[test]
    fn test_dcr_config() {
        let node = create_test_node(4, None, 3, Some("DCR"));
        let mut node = node;
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DepthToSpaceConfig>();

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::DCR);
    }

    #[test]
    fn test_crd_config() {
        let node = create_test_node(4, None, 3, Some("CRD"));
        let mut node = node;
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<DepthToSpaceConfig>();

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::CRD);
    }

    #[test]
    fn test_static_shape_update_outputs() {
        let mut node = create_test_node(4, Some(vec![2, 4, 2, 3]), 2, None);
        let processor = DepthToSpaceProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, vec![2, 1, 4, 6].into());
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
