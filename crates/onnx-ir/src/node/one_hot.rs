use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::NodeProcessor;
use std::any::Any;

/// Represents either a static value or a runtime argument for OneHot depth.
#[derive(Debug, Clone)]
pub enum OneHotDepthInput {
    /// Static depth known at compile time.
    Static(usize),
    /// Runtime depth determined during execution.
    Runtime(crate::ir::Argument),
}

/// Represents either a static value or a runtime argument for OneHot on/off values.
#[derive(Debug, Clone)]
pub enum OneHotValuesInput {
    /// Static values known at compile time.
    Static([f32; 2]),
    /// Runtime values determined during execution.
    Runtime(crate::ir::Argument),
}

/// Configuration for OneHot operation
#[derive(Debug, Clone)]
pub struct OneHotConfig {
    pub depth: OneHotDepthInput,
    pub values: OneHotValuesInput,
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
    fn process_config(&self, node: &mut Node, _opset: usize) {
        let depth = match node.inputs[1].into_value() {
            None => {
                // Runtime input - no static value available
                let mut runtime_arg = node.inputs[1].clone();
                runtime_arg.value_store = None;
                OneHotDepthInput::Runtime(runtime_arg)
            }
            Some(tensor_data) => {
                let depth_value = tensor_data.data.into_i64();
                OneHotDepthInput::Static(depth_value as usize)
            }
        };

        let values = match node.inputs[2].into_value() {
            None => {
                // Runtime input - no static value available
                let mut runtime_arg = node.inputs[2].clone();
                runtime_arg.value_store = None;
                OneHotValuesInput::Runtime(runtime_arg)
            }
            Some(tensor_data) => {
                let values_vec = tensor_data.data.into_f32s();
                let values_array: [f32; 2] = values_vec
                    .try_into()
                    .expect("OneHot: values must contain exactly 2 elements [off_value, on_value]");
                OneHotValuesInput::Static(values_array)
            }
        };

        let axis = node
            .attrs
            .get("axis")
            .map(|val| val.clone().into_i64())
            .unwrap_or(-1);

        let config = OneHotConfig {
            depth,
            values,
            axis,
        };
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
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
        let node = create_test_node(5, vec![0.0, 1.0], None).build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Static(d) if *d == 5));
        assert!(matches!(&config.values, OneHotValuesInput::Static(v) if v == &[0.0, 1.0]));
        assert_eq!(config.axis, -1); // default axis
    }

    #[test]
    fn test_one_hot_config_with_axis() {
        let node = create_test_node(5, vec![0.0, 1.0], Some(1)).build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Static(d) if *d == 5));
        assert!(matches!(&config.values, OneHotValuesInput::Static(v) if v == &[0.0, 1.0]));
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_one_hot_config_custom_values() {
        let node = create_test_node(10, vec![-1.0, 2.0], None).build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Static(d) if *d == 10));
        assert!(matches!(&config.values, OneHotValuesInput::Static(v) if v == &[-1.0, 2.0])); // custom off/on values
        assert_eq!(config.axis, -1);
    }

    #[test]
    fn test_one_hot_config_runtime_depth() {
        // Create node without registering depth constant in GraphData (runtime)
        let node = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", None) // No depth value (runtime)
            .input_tensor_f32_data("values", vec![0.0, 1.0], vec![2])
            .output_tensor_f32("output", 3, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Runtime(arg) if arg.name == "depth"));
        assert!(matches!(&config.values, OneHotValuesInput::Static(v) if v == &[0.0, 1.0]));
    }

    #[test]
    fn test_one_hot_config_runtime_values() {
        // Create node without registering values constant in GraphData (runtime)
        let node = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", Some(5))
            .input_tensor_f32("values", 1, None) // No values data (runtime)
            .output_tensor_f32("output", 3, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Static(d) if *d == 5));
        assert!(matches!(&config.values, OneHotValuesInput::Runtime(arg) if arg.name == "values"));
    }

    #[test]
    fn test_one_hot_config_both_runtime() {
        // Both depth and values are runtime
        let node = NodeBuilder::new(NodeType::OneHot, "test_one_hot")
            .input_tensor_i64("indices", 2, None)
            .input_scalar_tensor_i64("depth", None) // Runtime
            .input_tensor_f32("values", 1, None) // Runtime
            .output_tensor_f32("output", 3, None)
            .build_with_graph_data(16);
        let mut node = node;
        let processor = OneHotProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Runtime(arg) if arg.name == "depth"));
        assert!(matches!(&config.values, OneHotValuesInput::Runtime(arg) if arg.name == "values"));
    }
}
