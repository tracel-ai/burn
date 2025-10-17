use crate::ir::{ArgType, Node, NodeConfig, RuntimeInputRef, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Represents either a static value or a runtime argument for OneHot depth.
#[derive(Debug, Clone)]
pub enum OneHotDepthInput {
    /// Static depth known at compile time.
    Static(usize),
    /// Runtime depth determined during execution.
    Runtime(RuntimeInputRef),
}

/// Represents either a static value or a runtime argument for OneHot on/off values.
#[derive(Debug, Clone)]
pub enum OneHotValuesInput {
    /// Static values known at compile time.
    Static([f32; 2]),
    /// Runtime values determined during execution.
    Runtime(RuntimeInputRef),
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
pub fn one_hot_output_shape(node: &mut Node) -> Result<(), ProcessError> {
    log::debug!("OneHot rank inference for node {}", node.name);

    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: "OneHot: invalid input type".to_string(),
            });
        }
    };
    log::debug!("OneHot input rank for {}: {}", node.name, input_rank);

    let output_rank = input_rank + 1;
    log::debug!("OneHot output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.outputs[0].ty.elem_type().clone(),
        rank: output_rank,
        static_shape: None,
    });

    Ok(())
}

pub struct OneHotProcessor;

impl NodeProcessor for OneHotProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        let mut lifted = Vec::new();

        // Lift depth (input 1) and values (input 2)
        if node.inputs.len() > 1 {
            lifted.push(node.inputs[1].name.clone());
        }
        if node.inputs.len() > 2 {
            lifted.push(node.inputs[2].name.clone());
        }

        Ok(lifted)
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 9)?;
        crate::util::validate_min_inputs(node, 3)?;

        // Update output shape
        one_hot_output_shape(node)?;

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let depth = match node.inputs[1].into_value() {
            None => {
                // Runtime input - no static value available
                OneHotDepthInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
            }
            Some(tensor_data) => {
                let depth_value = tensor_data.data.into_i64();
                OneHotDepthInput::Static(depth_value as usize)
            }
        };

        let values = match node.inputs[2].into_value() {
            None => {
                // Runtime input - no static value available
                OneHotValuesInput::Runtime(RuntimeInputRef::new(node.inputs[2].name.clone(), 2))
            }
            Some(tensor_data) => {
                let values_vec = tensor_data.data.into_f32s();
                let values_array: [f32; 2] = values_vec.try_into().map_err(|_| {
                    ProcessError::Custom(
                        "OneHot: values must contain exactly 2 elements [off_value, on_value]"
                            .to_string(),
                    )
                })?;
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
        Ok(Some(Box::new(config)))
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
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
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<OneHotConfig>();
        assert!(matches!(&config.depth, OneHotDepthInput::Runtime(arg) if arg.name == "depth"));
        assert!(matches!(&config.values, OneHotValuesInput::Runtime(arg) if arg.name == "values"));
    }
}
