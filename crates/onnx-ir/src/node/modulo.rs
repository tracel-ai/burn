//! # Mod
//!
//! Element-wise binary modulus operation with Numpy-style broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Mod.html>
//!
//! ## Attributes
//! - `fmod` (int, default=0): Whether to use fmod (C-style) or integer modulo (Python-style)
//!   - `0` (default): Integer modulo - sign follows divisor (Python `%` operator)
//!   - `1`: Floating-point modulo - follows C `fmod` function, sign follows dividend
//!
//! ## Inputs
//! - `A` (T): Dividend tensor
//! - `B` (T): Divisor tensor
//!
//! ## Outputs
//! - `C` (T): Remainder tensor
//!
//! ## Opset Versions
//! - Since opset 10: Initial implementation
//! - Opset 13: Added support for additional types (bfloat16)

use crate::ir::{AttributeValue, Node, NodeConfig};
use crate::processor::{InputPreferences, NodeProcessor, OutputPreferences, ProcessError};

use std::any::Any;

/// Configuration for Mod operations
#[derive(Debug, Clone)]
pub struct ModConfig {
    /// Determines the modulo operation behavior:
    /// false (default): Integer modulo - sign follows divisor (Python-style %)
    /// true: Floating-point modulo (C-style fmod) - sign follows dividend
    pub fmod: bool,
}

impl ModConfig {
    /// Create a new ModConfig
    pub fn new(fmod: bool) -> Self {
        Self { fmod }
    }
}

impl NodeConfig for ModConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ModuloProcessor;

impl NodeProcessor for ModuloProcessor {
    fn input_preferences(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() != 2 {
            return Ok(None);
        }

        let mut prefs = InputPreferences::new();

        // Type propagation for Shape arithmetic (same as Add/Sub/Mul/Div)
        // Case 1: Shape op Constant => prefer Constant as Shape
        if node.inputs[0].ty.is_shape() {
            prefs = prefs.add(&node.inputs[1].name, ArgPreference::Shape);
        }

        // Case 2: Constant op Shape => prefer Constant as Shape
        if node.inputs[1].ty.is_shape() {
            prefs = prefs.add(&node.inputs[0].name, ArgPreference::Shape);
        }

        // Type propagation for Scalar arithmetic
        // Case 3: Scalar op Constant => prefer Constant as Scalar
        if node.inputs[0].ty.is_scalar() {
            prefs = prefs.add(&node.inputs[1].name, ArgPreference::Scalar);
        }

        // Case 4: Constant op Scalar => prefer Constant as Scalar
        if node.inputs[1].ty.is_scalar() {
            prefs = prefs.add(&node.inputs[0].name, ArgPreference::Scalar);
        }

        Ok(Some(prefs))
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 10)?;

        // FIXME: Should validate exactly 2 inputs, not minimum 2 (spec says A and B only)
        crate::processor::validate_min_inputs(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Add validation that fmod attribute, if present, is either 0 or 1

        // Output type is same as input with broadcasting
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract fmod attribute
        let fmod = match node.attrs.get("fmod") {
            Some(AttributeValue::Int64(value)) => *value != 0,
            _ => false, // Default value as per ONNX spec
        };

        let config = ModConfig::new(fmod);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node() -> crate::ir::Node {
        NodeBuilder::new(NodeType::Mod, "test_mod")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_f32("result", 2, None)
            .build()
    }

    #[test]
    fn test_mod_config_default() {
        let node = create_test_node();
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ModConfig>();
        assert_eq!(config.fmod, false); // Should default to false
    }

    #[test]
    fn test_mod_config_with_fmod_0() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(0));
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ModConfig>();
        assert_eq!(config.fmod, false);
    }

    #[test]
    fn test_mod_config_with_fmod_1() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(1));
        let mut node = node;
        let processor = ModuloProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<ModConfig>();
        assert_eq!(config.fmod, true);
    }
}
