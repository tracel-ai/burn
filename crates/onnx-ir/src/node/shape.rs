use crate::ir::{ArgType, Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for the Shape operation.
#[derive(Debug, Clone)]
pub struct ShapeConfig {
    pub start: usize,
    pub end: usize,
}

impl NodeConfig for ShapeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ShapeProcessor;

impl NodeProcessor for ShapeProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        if opset < 1 {
            return Err(ProcessError::UnsupportedOpset {
                required: 1,
                actual: opset,
            });
        }

        // Validate input count
        if node.inputs.len() != 1 {
            return Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: node.inputs.len(),
            });
        }

        // Validate output count
        if node.outputs.len() != 1 {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        // Extract the rank/dimension count from the input
        let rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            ArgType::Shape(rank) => *rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract attributes
        let mut start_dim: i64 = 0;
        let mut end_dim: i64 = rank as i64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "start" => start_dim = value.clone().into_i64(),
                "end" => end_dim = value.clone().into_i64(),
                _ => {}
            }
        }

        // Handle negative indices
        if start_dim < 0 {
            start_dim += rank as i64;
        }
        if end_dim < 0 {
            end_dim += rank as i64;
        }

        // Calculate dimensions
        let start = start_dim as usize;
        let end = end_dim as usize;
        let dim = end - start;

        // Store config for extract_config
        let config = ShapeConfig { start, end };
        node.config = Some(Box::new(config));

        // Infer output type - Shape always outputs Shape type
        node.outputs[0].ty = ArgType::Shape(dim);

        log::debug!("Shape node '{}': outputting Shape({})", node.name, dim);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the rank/dimension count from the input
        let rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            ArgType::Shape(rank) => *rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract attributes
        let mut start_dim: i64 = 0;
        let mut end_dim: i64 = rank as i64;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "start" => start_dim = value.clone().into_i64(),
                "end" => end_dim = value.clone().into_i64(),
                _ => {}
            }
        }

        // Handle negative indices
        if start_dim < 0 {
            start_dim += rank as i64;
        }
        if end_dim < 0 {
            end_dim += rank as i64;
        }

        // Calculate dimensions
        let start = start_dim as usize;
        let end = end_dim as usize;

        let config = ShapeConfig { start, end };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(start: Option<i64>, end: Option<i64>, rank: usize) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Shape, "test_shape")
            .input_tensor_f32("data", rank, None)
            .output_tensor_i64("shape", 1, None);

        if let Some(start_val) = start {
            builder = builder.attr_int("start", start_val);
        }

        if let Some(end_val) = end {
            builder = builder.attr_int("end", end_val);
        }

        builder.build()
    }

    #[test]
    fn test_shape_config_defaults() {
        let mut node = create_test_node(None, None, 4);
        let processor = ShapeProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ShapeConfig>();
        assert_eq!(config.start, 0);
        assert_eq!(config.end, 4);

        // Should always output Shape
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(4)));
    }

    #[test]
    fn test_shape_config_with_start() {
        let mut node = create_test_node(Some(1), None, 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ShapeConfig>();
        assert_eq!(config.start, 1);
        assert_eq!(config.end, 4);
    }

    #[test]
    fn test_shape_config_with_end() {
        let mut node = create_test_node(None, Some(3), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ShapeConfig>();
        assert_eq!(config.start, 0);
        assert_eq!(config.end, 3);
    }

    #[test]
    fn test_shape_config_with_start_and_end() {
        let mut node = create_test_node(Some(1), Some(3), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ShapeConfig>();
        assert_eq!(config.start, 1);
        assert_eq!(config.end, 3);
    }

    #[test]
    fn test_shape_config_negative_dims() {
        let mut node = create_test_node(Some(-2), Some(-1), 4);

        let processor = ShapeProcessor;

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ShapeConfig>();
        assert_eq!(config.start, 2); // -2 + 4 = 2
        assert_eq!(config.end, 3); // -1 + 4 = 3
    }

    #[test]
    fn test_shape_config_multiple_inputs() {
        let mut node = create_test_node(None, None, 4);
        // Add an extra input to cause error
        node.inputs.push(crate::ir::Argument {
            name: "extra".to_string(),
            ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                elem_type: crate::ir::ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value_store: None,
        });

        let processor = ShapeProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_shape_output_type() {
        // Shape operation always outputs Shape type
        let mut node = NodeBuilder::new(NodeType::Shape, "test_shape")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("shape", 1, None)
            .build();

        let processor = ShapeProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should always output Shape type
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(3)));
    }
}
