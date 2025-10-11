use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};
use std::any::Any;

/// Configuration for OneHot operation
#[derive(Debug, Clone)]
pub struct OneHotConfig {
    pub depth: usize,
    pub values: [f32; 2],
    pub axis: i64,
}

impl NodeConfig for OneHotConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub fn one_hot_config(curr: &Node, graph_data: &mut crate::from_onnx::GraphData) -> OneHotConfig {
    let depth = curr.inputs[1]
        .into_value()
        .clone()
        .expect("OneHot: Only constant depth is currently supported")
        .data
        .into_i64();

    let values = curr.inputs[2]
        .into_value()
        .clone()
        .expect("OneHot: Only constant on/off values is currently supported")
        .data
        .into_f32s();

    let axis = curr
        .attrs
        .get("axis")
        .map(|val| val.clone().into_i64())
        .unwrap_or(-1);

    OneHotConfig {
        depth: depth as usize,
        values: values.try_into().unwrap(),
        axis,
    }
}

/// Update output rank for OneHot (input rank + 1).
pub fn one_hot_output_shape(node: &mut Node) {
    log::debug!("OneHot rank inference for node {}", node.name);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("OneHot: invalid input type"),
    };
    log::debug!("OneHot input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank + 1;
    log::debug!("OneHot output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.outputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });
}

pub struct OneHotProcessor;

impl NodeProcessor for OneHotProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (9, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let config = one_hot_config(node, graph_data);
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        crate::node::one_hot::one_hot_output_shape(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(depth: i64, values: Vec<f32>, axis: Option<i64>) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", Some(depth))
            .input_tensor_f32_data("values", values.clone(), vec![2]) // always [off_value, on_value]
            .output_tensor_f32("output", 3, None); // rank increases by 1

        if let Some(axis_val) = axis {
            builder = builder.attr_int("axis", axis_val);
        }

        builder
    }

    #[test]
    fn test_one_hot_config_basic() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(5, vec![0.0, 1.0], None).build_with_graph_data(&mut graph_data);
        let config = one_hot_config(&node, &mut graph_data);
        assert_eq!(config.depth, 5);
        assert_eq!(config.values, [0.0, 1.0]);
        assert_eq!(config.axis, -1); // default axis
    }

    #[test]
    fn test_one_hot_config_with_axis() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node =
            create_test_node(5, vec![0.0, 1.0], Some(1)).build_with_graph_data(&mut graph_data);
        let config = one_hot_config(&node, &mut graph_data);
        assert_eq!(config.depth, 5);
        assert_eq!(config.values, [0.0, 1.0]);
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_one_hot_config_custom_values() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node =
            create_test_node(10, vec![-1.0, 2.0], None).build_with_graph_data(&mut graph_data);
        let config = one_hot_config(&node, &mut graph_data);
        assert_eq!(config.depth, 10);
        assert_eq!(config.values, [-1.0, 2.0]); // custom off/on values
        assert_eq!(config.axis, -1);
    }

    #[test]
    #[should_panic(expected = "Only constant depth is currently supported")]
    fn test_one_hot_config_no_depth_value() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        // Create node without registering depth constant in GraphData
        let node = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", None) // No depth value
            .input_tensor_f32_data("values", vec![0.0, 1.0], vec![2])
            .output_tensor_f32("output", 3, None)
            .build_with_graph_data(&mut graph_data);
        let _ = one_hot_config(&node, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "Only constant on/off values is currently supported")]
    fn test_one_hot_config_no_values() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        // Create node without registering values constant in GraphData
        let node = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", Some(5))
            .input_tensor_f32("values", 1, None) // No values data
            .output_tensor_f32("output", 3, None)
            .build_with_graph_data(&mut graph_data);
        let _ = one_hot_config(&node, &mut graph_data);
    }
}
