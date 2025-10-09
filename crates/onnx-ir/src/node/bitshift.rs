use crate::ir::Node;
use crate::processor::{NodeProcessor, ProcessorContext};

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
pub fn bitshift_config(node: &Node, graph_data: &mut crate::from_onnx::GraphData) -> Direction {
    let direction_str = node
        .attrs
        .get("direction")
        .map(|val| val.clone().into_string())
        .unwrap_or_else(|| "left".to_string());

    Direction::from_str(&direction_str)
        .unwrap_or_else(|e| panic!("Failed to parse bitshift direction: {e}"))
}

pub struct BitShiftProcessor;

impl NodeProcessor for BitShiftProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (11, None)
    }

    fn process(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
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

        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = bitshift_config(&node, &mut graph_data);
        assert_eq!(config, Direction::Left);
    }

    #[test]
    fn test_bitshift_config_with_direction_right() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "right")
            .build();

        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = bitshift_config(&node, &mut graph_data);
        assert_eq!(config, Direction::Right);
    }

    #[test]
    fn test_bitshift_config_default_direction() {
        let node = NodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .build();

        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = bitshift_config(&node, &mut graph_data);
        assert_eq!(config, Direction::Left);
    }
}
