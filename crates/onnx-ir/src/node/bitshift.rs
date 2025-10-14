use crate::ir::{Node, NodeConfig};
use crate::processor::NodeProcessor;
use crate::util::validate_opset;

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
    fn process_config(&self, node: &mut Node, opset: usize) {
        // BitShift implementation supports opset 11+
        validate_opset(&node.node_type, opset, 11);

        let direction_str = node
            .attrs
            .get("direction")
            .map(|val| val.clone().into_string())
            .unwrap_or_else(|| "left".to_string());

        let direction = Direction::from_str(&direction_str)
            .unwrap_or_else(|e| panic!("Failed to parse bitshift direction: {e}"));

        let config = BitShiftConfig { direction };
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        crate::util::same_as_input_broadcast(node);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
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
        processor.process_config(&mut node, 16);
        let config = node.config::<BitShiftConfig>();
        assert_eq!(config.direction, Direction::Left);
    }
}
