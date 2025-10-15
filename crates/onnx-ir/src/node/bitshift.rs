use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use std::any::Any;

pub use self::Direction as BitShiftDirection;

/// Direction for BitShift operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

impl Direction {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "left" => Ok(Direction::Left),
            "right" => Ok(Direction::Right),
            _ => Err(format!("Invalid bit shift direction: {s}")),
        }
    }
}

/// Configuration for BitShift operation
#[derive(Debug, Clone)]
pub struct BitShiftConfig {
    pub direction: Direction,
}

impl NodeConfig for BitShiftConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct BitShiftProcessor;

impl NodeProcessor for BitShiftProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        if opset < 11 {
            return Err(ProcessError::UnsupportedOpset {
                required: 11,
                actual: opset,
            });
        }

        // Validate input count (need at least 2 inputs)
        if node.inputs.len() < 2 {
            return Err(ProcessError::InvalidInputCount {
                expected: 2,
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

        // Extract direction attribute
        let direction_str = node
            .attrs
            .get("direction")
            .map(|val| val.clone().into_string())
            .unwrap_or_else(|| "left".to_string());

        let direction =
            Direction::from_str(&direction_str).map_err(|e| ProcessError::InvalidAttribute {
                name: "direction".to_string(),
                reason: e,
            })?;

        let config = BitShiftConfig { direction };
        node.config = Some(Box::new(config));

        // Output type is same as input with broadcasting
        crate::util::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        Ok(node.config.as_ref().map(|c| c.clone_box()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    #[test]
    fn test_bitshift_config_with_direction_left() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "left")
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BitShiftConfig>();
        assert_eq!(config.direction, Direction::Left);
    }

    #[test]
    fn test_bitshift_config_with_direction_right() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "right")
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BitShiftConfig>();
        assert_eq!(config.direction, Direction::Right);
    }

    #[test]
    fn test_bitshift_config_default_direction() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BitShiftConfig>();
        assert_eq!(config.direction, Direction::Left);
    }
}
