use crate::processor::NodeProcessor;

use crate::ir::{ArgType, Data, Node, NodeConfig, TensorType};
use std::any::Any;

/// Represents either a static value or a runtime argument for squeeze axes.
#[derive(Debug, Clone)]
pub enum SqueezeInput {
    /// Static axes known at compile time.
    Static(Vec<i64>),
    /// Runtime axes determined during execution.
    Runtime(crate::ir::Argument),
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
    fn process_config(&self, node: &mut Node, _opset: usize) {
        fn get_squeeze_axes(node: &Node) -> Option<SqueezeInput> {
            // In ONNX opset 13+, axes are provided as a second input
            if node.inputs.len() < 2 {
                return None; // No axes input means squeeze all dims with size 1
            }

            let input = &node.inputs[1];
            match input.into_value() {
                None => {
                    // Runtime input - no static value available
                    let mut runtime_arg = input.clone();
                    runtime_arg.value_store = None;
                    Some(SqueezeInput::Runtime(runtime_arg))
                }
                Some(value) => match &value.data {
                    Data::Int64s(axes) => Some(SqueezeInput::Static(axes.clone())),
                    _ => panic!("Squeeze: axes must be int64 type"),
                },
            }
        }

        let axes = get_squeeze_axes(node);
        let config = SqueezeConfig { axes };
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        log::debug!("Squeeze rank inference for node {}", node.name);

        // Extract axes from config
        let config = node.config::<SqueezeConfig>();
        // TODO mark axes input as Shape input if constant
        let axes = match &config.axes {
            Some(SqueezeInput::Static(axes_vec)) => Some(axes_vec.clone()),
            Some(SqueezeInput::Runtime(_)) => {
                // Runtime axes - cannot infer rank without static information
                None
            }
            None => None,
        };

        log::debug!("Squeeze axes for {}: {:?}", node.name, axes);

        match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                log::debug!("Squeeze input rank for {}: {}", node.name, tensor.rank);
                let output_rank = match axes {
                    None => {
                        // When axes is None, ONNX spec squeezes all dimensions of size 1
                        // Without static shape info, we can't know which dims are size 1
                        // The output type will be corrected later if ONNX provides it
                        // TODO: Infer rank from output tensor shape based on static shape inference
                        if let Some(ref static_shape) = tensor.static_shape {
                            // Count the number of dimensions not equal to 1
                            static_shape.iter().filter(|&&dim| dim != 1).count()
                        } else {
                            panic!(
                                "Squeeze: Cannot infer output rank when axes is None and input tensor static shape is unknown. Please provide static shape information for accurate inference."
                            );
                        }
                    }
                    Some(ref axes_vec) => tensor.rank - axes_vec.len(),
                };
                log::debug!("Squeeze output rank for {}: {}", node.name, output_rank);

                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
            }
            ArgType::Shape(shape_rank) => {
                log::debug!("Squeeze input is Shape({}) for {}", shape_rank, node.name);

                // Shape is always a 1D array. We can only squeeze axis 0.
                // - If Shape has 1 element (Shape(1)), squeezing axis 0 produces a scalar
                // - If Shape has >1 elements (Shape(n) where n>1), squeezing axis 0 is a no-op
                //   because the dimension has size > 1

                if let Some(ref axes_vec) = axes
                    && !axes_vec.is_empty()
                    && (axes_vec.len() != 1 || axes_vec[0] != 0)
                {
                    panic!(
                        "Squeeze on Shape input only supports squeezing axis 0, got axes: {axes_vec:?}"
                    );
                }

                if *shape_rank == 1 {
                    // Shape(1) squeezed on axis 0 produces a scalar
                    node.outputs[0].ty = ArgType::Scalar(crate::ir::ElementType::Int64);
                    log::debug!("Squeeze Shape(1) to Scalar for {}", node.name);
                } else {
                    // Shape(n) where n > 1 remains unchanged
                    node.outputs[0].ty = ArgType::Shape(*shape_rank);
                    log::debug!("Squeeze Shape({}) unchanged for {}", shape_rank, node.name);
                }
            }
            ArgType::Scalar(scalar_type) => {
                // Scalar squeeze is a no-op
                node.outputs[0].ty = ArgType::Scalar(scalar_type.clone());
                log::debug!("Squeeze Scalar unchanged for {}", node.name);
            }
        }
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
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64("axes", 0, None) // Runtime input - no static value
            .output_tensor_f32("squeezed", 2, None)
    }

    #[test]
    fn test_squeeze_config_with_axes_input() {
        let node = create_test_node(Some(vec![0, 2]), 4).build_with_graph_data(16);
        let mut node = node;
        let processor = SqueezeProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<SqueezeConfig>();
        assert!(matches!(config.axes, Some(SqueezeInput::Static(ref axes)) if axes == &vec![0, 2]));
    }

    #[test]
    fn test_squeeze_config_no_axes_input() {
        let node = create_test_node(None, 4).build();
        let mut node = node;
        let processor = SqueezeProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<SqueezeConfig>();
        assert!(config.axes.is_none());
    }

    #[test]
    fn test_squeeze_config_runtime_axes() {
        let node = create_runtime_squeeze_node().build();
        let mut node = node;
        let processor = SqueezeProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<SqueezeConfig>();
        assert!(matches!(config.axes, Some(SqueezeInput::Runtime(ref arg)) if arg.name == "axes"));
    }
}
