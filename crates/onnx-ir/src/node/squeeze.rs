//! # Squeeze
//!
//! Removes single-dimensional entries from the shape of a tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Squeeze.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with optional 'axes' attribute.
//! - **Opset 11**: Clarified semantics and behavior for negative axis values.
//! - **Opset 13**: Changed 'axes' from attribute to optional input, enabling dynamic axes specification at runtime.
//!
//! **Implementation Note**: This implementation requires opset 13+ (axes as input). The change from attribute to input provides greater flexibility for dynamic shape operations.

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use crate::ir::{ArgType, Node, NodeBuilder, RuntimeInputRef, TensorDataExt, TensorType};

/// Represents either a static value or a runtime argument for squeeze axes.
#[derive(Debug, Clone)]
pub enum SqueezeInput {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes determined during execution.
    Runtime(RuntimeInputRef),
}

impl Default for SqueezeInput {
    fn default() -> Self {
        SqueezeInput::Static(vec![])
    }
}

/// Configuration for Squeeze operation
#[derive(Debug, Clone)]
pub struct SqueezeConfig {
    pub axes: Option<SqueezeInput>,
}

pub(crate) struct SqueezeProcessor;

impl NodeProcessor for SqueezeProcessor {
    type Config = SqueezeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 13,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Lift axes input (input[1]) if present
        // FIXME: This should check if the input is constant before attempting to lift,
        // similar to other processors. Currently it lifts unconditionally if present.
        // Should use: if node.inputs[1].is_constant() { node.inputs[1].to_static()?; }
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Get reference to config for type inference
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");
        let axes = config.axes.clone();

        // Extract axes for type inference
        let axes_vec = match &axes {
            Some(SqueezeInput::Static(axes_vec)) => Some(axes_vec.clone()),
            Some(SqueezeInput::Runtime(_)) => None,
            None => None,
        };

        // TODO: Missing validation that axes values are in valid range [-rank, rank-1].
        // Out-of-bounds axes should be rejected but aren't validated here.

        // TODO: Missing validation that axes doesn't contain duplicates.
        // Duplicate axes should be rejected per ONNX spec but not validated.

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

                        // TODO: Missing validation that squeezed dimensions actually have size 1.
                        // ONNX spec requires dimensions to be size 1 to be squeezed, but implementation
                        // doesn't validate this when static_shape is available. Should check:
                        // for &axis in axes_vec { assert static_shape[axis] == 1 }

                        tensor.rank - axes_vec.len()
                    }
                };

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    dtype: tensor.dtype,
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
                    node.outputs[0].ty = ArgType::Scalar(crate::ir::DType::I64);
                } else {
                    node.outputs[0].ty = ArgType::Shape(*shape_rank);
                }
            }
            ArgType::Scalar(scalar_type) => {
                node.outputs[0].ty = ArgType::Scalar(*scalar_type);
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        fn get_squeeze_axes(node: &NodeBuilder) -> Option<SqueezeInput> {
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
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Squeeze {
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

    fn create_test_node(axes: Option<Vec<i64>>, rank: usize) -> TestNodeBuilder {
        let output_rank = if let Some(ref axes_vec) = axes {
            rank - axes_vec.len()
        } else {
            // When no axes specified, we don't know how many dims will be squeezed
            // without static shape info, but for testing we'll assume same as input
            rank
        };

        let mut builder = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("squeezed", output_rank, None);

        // Add axes as a second input (ONNX opset 13+ style)
        if let Some(axes_val) = axes {
            builder = builder.input_tensor_i64_data("axes", axes_val.clone(), vec![axes_val.len()]);
        }

        builder
    }

    fn create_runtime_squeeze_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Squeeze, "test_runtime_squeeze")
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
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(config.axes, Some(SqueezeInput::Static(ref axes)) if axes == &vec![0, 2]));
    }

    #[test]
    fn test_squeeze_config_no_axes_input() {
        // Test with no axes input - need static shape with dims of size 1
        let node = TestNodeBuilder::new(NodeType::Squeeze, "test_squeeze")
            .input_tensor_f32("data", 4, Some(vec![2, 1, 3, 1])) // Has two dims of size 1
            .output_tensor_f32("squeezed", 2, None) // Will squeeze to rank 2
            .build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(config.axes.is_none());
    }

    #[test]
    fn test_squeeze_config_runtime_axes() {
        let node = create_runtime_squeeze_node().build();
        let mut node = node;
        let processor = SqueezeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(matches!(config.axes, Some(SqueezeInput::Runtime(ref arg)) if arg.name == "axes"));
    }

    // TODO: Missing test for squeezing dimension that is not size 1 - should fail.
    // E.g., input shape [2, 1, 3], axes=[0] should fail because dim 0 has size 2, not 1.

    // TODO: Missing test for negative axes normalization and validation.
    // E.g., axes=[-1] for rank-3 should squeeze last dimension.

    // TODO: Missing test for duplicate axes - axes=[0, 0] should be rejected.

    // TODO: Missing test for out-of-bounds axes - axes=[5] for rank-3 should be rejected.

    // TODO: Missing test for opset < 13 behavior - axes as attribute vs input.
    // Implementation requires opset 13+ but this transition isn't tested.

    // TODO: Missing test for squeezing all dimensions to create rank-0 tensor (scalar).
    // E.g., input shape [1, 1, 1] with no axes should result in scalar.
}
