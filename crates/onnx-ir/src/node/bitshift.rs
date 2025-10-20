//! # BitShift
//!
//! Performs element-wise bitwise shift operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BitShift.html>
//!
//! ## Attributes
//! - `direction` (string, required): Direction of moving bits. Can be either "RIGHT" (for right shift)
//!   or "LEFT" (for left shift). When direction is "RIGHT", the operator moves the binary
//!   representation toward the right side, effectively decreasing the input value. When direction
//!   is "LEFT", bits move toward the left side, increasing the actual value.
//!   Note: Implementation provides default "left" despite spec marking as required.
//!
//! ## Inputs
//! - `X` (T, required): Tensor to be shifted
//! - `Y` (T, required): Tensor specifying the amounts of shifting (number of bits to shift)
//!
//! ## Outputs
//! - `Z` (T): Output tensor with shifted values
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
        crate::processor::validate_opset(opset, 11)?;
        crate::processor::validate_min_inputs(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Add validation for unexpected attributes
        // FIXME: Spec says 'direction' is required but extract_config provides default "left"
        // Should either validate presence here or update spec documentation

        // Output type is same as input with broadcasting
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
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
        Ok(Some(Box::new(config)))
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
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
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<BitShiftConfig>();
        assert_eq!(config.direction, Direction::Left);
    }
}
