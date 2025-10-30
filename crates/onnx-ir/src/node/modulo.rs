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
//! - **Opset 10-12**: Initial implementation with fmod attribute
//! - **Opset 13+**: Extended type support (added bfloat16)
//!
//! ## Missing Test Coverage
//! - TODO: No test for fmod values other than 0 or 1 - Spec only defines 0 and 1, other values should be rejected
//! - TODO: No test for dtype validation - Should ensure both inputs have compatible numeric types
//! - TODO: No test for zero divisor - Division by zero handling not tested
//! - TODO: No test for negative divisors with both fmod modes - Sign handling edge cases
//! - TODO: No test for integer types - Spec supports int8, int16, int32, int64, uint8, uint16, uint32, uint64
//! - TODO: No test for mixed sign operands - fmod=0 vs fmod=1 produces different results

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

        // Mod requires exactly 2 inputs: A and B
        crate::processor::validate_input_count(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Validate input dtypes are numeric - Integer and floating-point types supported - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Validate both inputs have same dtype - Mixed types should be rejected - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Add validation that fmod attribute, if present, is either 0 or 1 - Other values are undefined - burn/crates/onnx-ir/src/node/modulo.rs:100

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
            Some(AttributeValue::Int64(value)) => {
                // TODO: Validate fmod is 0 or 1 - Values other than 0 or 1 are undefined in spec - burn/crates/onnx-ir/src/node/modulo.rs:120
                *value != 0
            }
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
