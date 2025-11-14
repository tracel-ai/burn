//! # Mod
//!
//! Element-wise binary modulus operation with Numpy-style broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Mod.html>
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

use crate::ir::{AttributeValue, Node, NodeBuilder};
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError,
};

/// Configuration for Mod operations
#[derive(Debug, Clone, Default)]
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

pub(crate) struct ModuloProcessor;

impl NodeProcessor for ModuloProcessor {
    type Config = ModConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &NodeBuilder,
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input dtypes are numeric - Integer and floating-point types supported - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Validate both inputs have same dtype - Mixed types should be rejected - burn/crates/onnx-ir/src/node/modulo.rs:100
        // TODO: Add validation that fmod attribute, if present, is either 0 or 1 - Other values are undefined - burn/crates/onnx-ir/src/node/modulo.rs:100

        // Output type is same as input with broadcasting
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract fmod attribute
        let fmod = match node.attrs.get("fmod") {
            Some(AttributeValue::Int64(value)) => {
                // TODO: Validate fmod is 0 or 1 - Values other than 0 or 1 are undefined in spec - burn/crates/onnx-ir/src/node/modulo.rs:120
                *value != 0
            }
            _ => false, // Default value as per ONNX spec
        };

        let config = ModConfig::new(fmod);
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Mod {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use super::*;
    use crate::ir::{AttributeValue, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node() -> crate::ir::NodeBuilder {
        TestNodeBuilder::new(NodeType::Mod, "test_mod")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.fmod, true);
    }
}
