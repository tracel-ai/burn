//! # BitShift
//!
//! Performs element-wise bitwise shift operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BitShift.html>
//!
//! ## Type Constraints
//! - T in (tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64))
//!
//! ## Broadcasting
//! This operator supports multidirectional (i.e., Numpy-style) broadcasting.
//!
//! ## Opset Versions
//! - **Opset 11**: Initial version with left/right bitwise shift operations on unsigned integers
//!
//! ## Examples
//! - If direction is "RIGHT", X = [1, 4], and Y = [1, 1], output Z = [0, 2]
//! - If direction is "LEFT", X = [1, 2], and Y = [1, 2], output Z = [2, 8]
use crate::ir::Argument;

use crate::ir::{Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Direction for BitShift operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    #[default]
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

/// Node representation for BitShift operation
#[derive(Debug, Clone)]
pub struct BitShiftNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: BitShiftConfig,
}

pub(crate) struct BitShiftProcessor;

impl NodeProcessor for BitShiftProcessor {
    type Config = BitShiftConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add validation for unexpected attributes
        // FIXME: Spec says 'direction' is required but extract_config provides default "left"
        // Should either validate presence here or update spec documentation

        // Output type is same as input with broadcasting
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract direction attribute
        // FIXME: Spec marks 'direction' as required, but we provide default "left"
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
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::BitShift(BitShiftNode {
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

    #[test]
    fn test_bitshift_config_with_direction_left() {
        let node = TestNodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "left")
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.direction, Direction::Left);
    }

    #[test]
    fn test_bitshift_config_with_direction_right() {
        let node = TestNodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .attr_string("direction", "right")
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.direction, Direction::Right);
    }

    #[test]
    fn test_bitshift_config_default_direction() {
        let node = TestNodeBuilder::new(NodeType::BitShift, "test_bitshift")
            .input_tensor_i32("X", 2, None)
            .input_tensor_i32("Y", 2, None)
            .output_tensor_i32("Z", 2, None)
            .build();

        let mut node = node;
        let processor = BitShiftProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.direction, Direction::Left);
    }
}
