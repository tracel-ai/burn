//! # OneHot
//!
//! Produces a one-hot encoded tensor from input indices.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__OneHot.html>
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version with indices, depth, and values inputs.
//! - **Opset 11**: Added support for negative axis values and clarified axis semantics.
//!
//! ## Type Constraints (from ONNX spec)
//! - T1 (indices): tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
//! - T2 (depth): tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
//! - T3 (values/output): tensor(float), tensor(int32), tensor(int64), tensor(float16), etc.
//!
//! TODO: Add type constraint validation for indices and depth inputs
//! Current implementation doesn't validate that indices and depth are integer types.
//! Should reject non-integer types like float for indices/depth inputs.
//! Location: infer_types method after validate_input_count

use crate::ir::{
    ArgType, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt, TensorType,
};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
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
pub fn one_hot_output_shape(node: &mut NodeBuilder) -> Result<(), ProcessError> {
    let input_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: "OneHot: invalid input type".to_string(),
            });
        }
    };

    let output_rank = input_rank + 1;

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        dtype: node.outputs[0].ty.elem_type(),
        rank: output_rank,
        static_shape: None,
    });

    Ok(())
}

pub struct OneHotProcessor;

impl NodeProcessor for OneHotProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Lift depth (input 1) and values (input 2)
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate that depth input is scalar or rank-1 tensor with single element as per spec
        // Current implementation extracts depth[0] in extract_config but doesn't validate shape.
        // Should add validation: depth must be scalar (rank 0) or rank-1 tensor with shape [1].
        // Location: After validate_input_count, before one_hot_output_shape

        // TODO: Validate that values input has exactly 2 elements [off_value, on_value]
        // The spec requires values to be a 2-element tensor, but this is only checked in extract_config
        // when static value is available. For runtime values, no validation occurs.
        // Should validate tensor shape is [2] when values is a tensor input.
        // Location: After validate_input_count, before one_hot_output_shape

        // TODO: Missing test coverage for negative indices handling
        // ONNX spec states: "If an index is less than 0, it is treated as -1 and the corresponding
        // output position is set to off_value". No test validates this edge case.
        // Add test: one_hot_negative_indices

        // TODO: Missing test coverage for out-of-bounds indices
        // ONNX spec states: "If index value is greater than or equal to depth, the output is filled
        // with off_value". No test validates this behavior.
        // Add test: one_hot_out_of_bounds_indices

        // TODO: Missing test coverage for negative axis values
        // Opset 11 added support for negative axis, but no test validates axis=-2, axis=-3, etc.
        // Add test: one_hot_negative_axis

        // Update output shape
        one_hot_output_shape(node)?;

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let depth = match node.inputs[1].value() {
            None => {
                // Runtime input - no static value available
                OneHotDepthInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
            }
            Some(tensor_data) => {
                let depth_value = tensor_data.as_slice::<i64>().unwrap()[0];
                OneHotDepthInput::Static(depth_value as usize)
            }
        };

        let values = match node.inputs[2].value() {
            None => {
                // Runtime input - no static value available
                OneHotValuesInput::Runtime(RuntimeInputRef::new(node.inputs[2].name.clone(), 2))
            }
            Some(tensor_data) => {
                // Convert to f32 regardless of the input type
                // Values should be a 2-element tensor [off_value, on_value]
                let values_vec = tensor_data.to_f32_vec().map_err(|_| {
                    ProcessError::Custom("OneHot: unsupported values type".to_string())
                })?;

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

    fn build_node(&self, builder: NodeBuilder) -> Node {
        let config = builder
            .config
            .expect("Config should be set by extract_config")
            .as_any()
            .downcast_ref::<OneHotConfig>()
            .expect("Wrong config type")
            .clone();

        Node::OneHot {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(depth: i64, values: Vec<f32>, axis: Option<i64>) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::OneHot, "test_one_hot")
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
        let node = TestNodeBuilder::new(NodeType::OneHot, "test_one_hot")
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
        let node = TestNodeBuilder::new(NodeType::OneHot, "test_one_hot")
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
        let node = TestNodeBuilder::new(NodeType::OneHot, "test_one_hot")
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
