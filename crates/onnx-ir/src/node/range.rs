use crate::ir::{ArgType, Data, ElementType, Node, NodeConfig, TensorData, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};
use std::any::Any;

/// Configuration for the Range operation.
#[derive(Debug, Clone)]
pub struct RangeConfig {
    pub start: RangeInput,
    pub limit: RangeInput,
    pub delta: RangeInput,
}

impl NodeConfig for RangeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for range parameters.
#[derive(Debug, Clone)]
pub enum RangeInput {
    /// Static value known at compile time.
    Static(i64),
    /// Runtime argument determined during execution (stores argument name).
    Runtime(String),
}

pub struct RangeProcessor;

impl NodeProcessor for RangeProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (11, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        fn get_range_input(
            node: &Node,
            index: usize,
            param_name: &str,
            graph_data: &mut crate::from_onnx::GraphData,
        ) -> RangeInput {
            let input = node
                .inputs
                .get(index)
                .unwrap_or_else(|| panic!("Range: {} parameter is required", param_name));

            match input.into_value() {
                None => RangeInput::Runtime(input.name.clone()),
                Some(TensorData {
                    data: Data::Int64s(values),
                    ..
                }) if values.len() == 1 => RangeInput::Static(values[0]),
                Some(TensorData {
                    data: Data::Int32s(values),
                    ..
                }) if values.len() == 1 => RangeInput::Static(values[0] as i64),
                Some(v) => panic!(
                    "Range {} must be a scalar int value, got {:?}",
                    param_name, v
                ),
            }
        }

        let start = get_range_input(node, 0, "start", graph_data);
        let limit = get_range_input(node, 1, "limit", graph_data);
        let delta = get_range_input(node, 2, "delta", graph_data);

        let config = RangeConfig {
            start,
            limit,
            delta,
        };
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        log::debug!("Range rank inference for node {}", node.name);

        if node.inputs.len() != 3 {
            panic!("Range: expected 3 inputs, found {}", node.inputs.len());
        }
        log::debug!(
            "Range operation always produces rank 1 tensor for {}",
            node.name
        );

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: None,
        });

        log::debug!("Range output rank for {}: 1", node.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node() -> Node {
        NodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_i64("start")
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None) // Rank 0 will be updated
            .build()
    }

    #[test]
    fn test_range_output() {
        let mut node = create_test_node();
        let processor = RangeProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Range: expected 3 inputs, found 2")]
    fn test_range_missing_inputs() {
        let mut node = create_test_node();
        node.inputs.pop();
        let processor = RangeProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process_forward(&mut node, &context, &mut graph_data);
    }
}
