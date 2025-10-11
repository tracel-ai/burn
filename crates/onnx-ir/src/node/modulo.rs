use crate::ir::{AttributeValue, Node, NodeConfig};
use crate::processor::{NodeProcessor, ProcessorContext};
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
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (10, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let fmod = match node.attrs.get("fmod") {
            Some(AttributeValue::Int64(value)) => *value != 0,
            _ => false, // Default value as per ONNX spec
        };

        let config = ModConfig::new(fmod);
        node.config = Some(Box::new(config));
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
        let mut node = node;
        let processor = ModuloProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ModConfig>()
            .unwrap();
        assert_eq!(config.fmod, false); // Should default to false
    }

    #[test]
    fn test_mod_config_with_fmod_0() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(0));
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = node;
        let processor = ModuloProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ModConfig>()
            .unwrap();
        assert_eq!(config.fmod, false);
    }

    #[test]
    fn test_mod_config_with_fmod_1() {
        let mut node = create_test_node();
        node.attrs
            .insert("fmod".to_string(), AttributeValue::Int64(1));
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = node;
        let processor = ModuloProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ModConfig>()
            .unwrap();
        assert_eq!(config.fmod, true);
    }
}
