use crate::ir::{AttributeValue, Node};
use crate::processor::{NodeProcessor, ProcessorContext};

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

/// Create a ModConfig from the node attributes
pub fn mod_config(
    node: &crate::ir::Node,
    _graph_data: &mut crate::from_onnx::GraphData,
) -> ModConfig {
    let fmod = match node.attrs.get("fmod") {
        Some(AttributeValue::Int64(value)) => *value != 0,
        _ => false, // Default value as per ONNX spec
    };
    ModConfig::new(fmod)
}

pub struct ModuloProcessor;

impl NodeProcessor for ModuloProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (10, None)
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        crate::util::same_as_input_broadcast(node);
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
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = mod_config(&node, &mut graph_data);
        assert_eq!(config.fmod, false); // Should default to false
    }

    #[test]
    fn test_mod_config_with_fmod_0() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(0));
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = mod_config(&node, &mut graph_data);
        assert_eq!(config.fmod, false);
    }

    #[test]
    fn test_mod_config_with_fmod_1() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(1));
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = mod_config(&node, &mut graph_data);
        assert_eq!(config.fmod, true);
    }
}
