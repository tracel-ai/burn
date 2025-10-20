//! # Squeeze
//!
//! Removes single-dimensional entries from the shape of a tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Squeeze.html>
//!
//! ## Attributes
//!
//! None in opset 13+. In earlier versions (opset 11 and below), `axes` was an attribute.
//!
//! ## Inputs
//!
//! - `data` (T): Tensor with at least max(axes) dimensions
//! - `axes` (tensor(int64), optional): List of integers indicating the dimensions to squeeze.
//!   Negative values count dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
//!   If not provided, all dimensions of size 1 will be removed.
//!
//! ## Outputs
//!
//! - `squeezed` (T): Reshaped tensor with same data as input, with specified dimensions of size 1 removed
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with optional 'axes' attribute.
//! - **Opset 11**: Clarified semantics and behavior for negative axis values.
//! - **Opset 13**: Changed 'axes' from attribute to optional input, enabling dynamic axes specification at runtime.
//!
//! **Implementation Note**: This implementation requires opset 13+ (axes as input). The change from attribute to input provides greater flexibility for dynamic shape operations.

use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use crate::ir::{ArgType, Node, NodeConfig, RuntimeInputRef, TensorType};
use std::any::Any;

/// Represents either a static value or a runtime argument for squeeze axes.
#[derive(Debug, Clone)]
pub enum SqueezeInput {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes determined during execution.
    Runtime(RuntimeInputRef),
}

/// Configuration for Squeeze operation
#[derive(Debug, Clone)]
pub struct SqueezeConfig {
    pub axes: Option<SqueezeInput>,
}

impl NodeConfig for SqueezeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct SqueezeProcessor;

impl NodeProcessor for SqueezeProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
        // FIXME: This should check if the input is constant before attempting to lift,
        // similar to other processors. Currently it lifts unconditionally if present.
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 13)?;

        // Validate we have at least one input
        crate::processor::validate_min_inputs(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Get reference to config for type inference
        let config = node.config::<SqueezeConfig>();
        let axes = config.axes.clone();

        // Extract axes for type inference
        let axes_vec = match &axes {
            Some(SqueezeInput::Static(axes_vec)) => Some(axes_vec.clone()),
            Some(SqueezeInput::Runtime(_)) => None,
            None => None,
        };

        match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                let output_rank = match axes_vec {
                    None => {
                        // When axes is None, ONNX spec squeezes all dimensions of size 1
                        if let Some(ref static_shape) = tensor.static_shape {
                            static_shape.iter().filter(|&&dim| dim != 1).count()
                        } else {
                            return Err(ProcessError::Custom(
                                "Squeeze: Cannot infer output rank when axes is None and input tensor static shape is unknown".to_string()
                            ));
                        }
                    }
                    Some(ref axes_vec) => {
                        // Validate that we're not trying to squeeze more axes than the tensor has
                        if axes_vec.len() > tensor.rank {
                            return Err(ProcessError::Custom(format!(
                                "Squeeze: Cannot squeeze {} axes from a rank {} tensor",
                                axes_vec.len(),
                                tensor.rank
                            )));
                        }
                        tensor.rank - axes_vec.len()
                    }
                };

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
            }
            ArgType::Shape(shape_rank) => {
                if let Some(ref axes_vec) = axes_vec
                    && !axes_vec.is_empty()
                    && (axes_vec.len() != 1 || axes_vec[0] != 0)
                {
                    return Err(ProcessError::Custom(format!(
                        "Squeeze on Shape input only supports squeezing axis 0, got axes: {:?}",
                        axes_vec
                    )));
                }

                if *shape_rank == 1 {
                    node.outputs[0].ty = ArgType::Scalar(crate::ir::ElementType::Int64);
                } else {
                    node.outputs[0].ty = ArgType::Shape(*shape_rank);
                }
            }
            ArgType::Scalar(scalar_type) => {
                node.outputs[0].ty = ArgType::Scalar(scalar_type.clone());
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        fn get_squeeze_axes(node: &Node) -> Option<SqueezeInput> {
            // In ONNX opset 13+, axes are provided as a second input
            if node.inputs.len() < 2 {
                return None; // No axes input means squeeze all dims with size 1
            }

            let input = &node.inputs[1];
            match input.value() {
                None => {
                    // Runtime input - no static value available
                    Some(SqueezeInput::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        1,
                    )))
                }
                Some(value) => match value.to_i64_vec() {
                    Ok(axes) => Some(SqueezeInput::Static(axes)),
                    Err(_) => None, // Invalid type
                },
            }
        }

        let axes = get_squeeze_axes(node);
        let config = SqueezeConfig { axes };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> NodeBuilder {
        let output_rank = if let Some(ref axes_vec) = axes {
            rank - axes_vec.len()
        } else {
            // When no axes specified, we don't know how many dims will be squeezed
            // without static shape info, but for testing we'll assume same as input
            rank
        };

        let mut builder = NodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("squeezed", output_rank, None);

        // Add axes as a second input (ONNX opset 13+ style)
        if let Some(axes_val) = axes {
            builder = builder.input_tensor_i64_data("axes", axes_val.clone(), vec![axes_val.len()]);
        }

        builder
    }

    fn create_runtime_squeeze_node() -> NodeBuilder {
        NodeBuilder::new(NodeType::Squeeze, "test_runtime_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 3, 4, 5])) // Need some shape
            .input_tensor_i64("axes", 0, None) // Runtime input - no static value
            .output_tensor_f32("squeezed", 2, None)
    }

    #[test]
    fn test_squeeze_config_with_axes_input() {
        let node = create_test_node(Some(vec![0, 2]), 4).build_with_graph_data(16);
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SqueezeConfig>();
        assert!(matches!(config.axes, Some(SqueezeInput::Static(ref axes)) if axes == &vec![0, 2]));
    }

    #[test]
    fn test_squeeze_config_no_axes_input() {
        // Test with no axes input - need static shape with dims of size 1
        let node = NodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 1, 3, 1])) // Has two dims of size 1
            .output_tensor_f32("squeezed", 2, None) // Will squeeze to rank 2
            .build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SqueezeConfig>();
        assert!(config.axes.is_none());
    }

    #[test]
    fn test_squeeze_config_runtime_axes() {
        let node = create_runtime_squeeze_node().build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SqueezeConfig>();
        assert!(matches!(config.axes, Some(SqueezeInput::Runtime(ref arg)) if arg.name == "axes"));
    }
}
