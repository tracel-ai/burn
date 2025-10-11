use crate::processor::NodeProcessor;
use crate::{ArgType, Node, NodeConfig, TensorType};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub dims: Vec<usize>,
    pub keepdims: bool,
}

impl NodeConfig for ReduceConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

impl ReduceConfig {
    pub fn new(dims: Vec<usize>, keepdims: bool) -> Self {
        Self { dims, keepdims }
    }
}

pub struct ReduceProcessor;

impl NodeProcessor for ReduceProcessor {
    fn process_config(&self, node: &mut Node, _opset: usize) {
        let mut axes = Vec::new();
        let mut keepdims = 1;

        let tensor = match node.inputs.first().unwrap().clone().ty {
            ArgType::Tensor(tensor) => tensor,
            _ => panic!("{}: Only tensor input is valid", node.node_type),
        };

        // Extract the attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axes" => axes = value.clone().into_i64s(),
                "keepdims" => keepdims = value.clone().into_i64(),
                _ => {}
            }
        }

        // Process axes from additional input (if available)
        if let Some(value) = node
            .inputs
            .get(1)
            .and_then(|argument| argument.into_value())
        {
            axes = value.data.into_i64s();
        }

        let mut dims: Vec<usize> = axes
            .into_iter()
            .map(|mut dim| {
                if dim < 0 {
                    // Accepted range is [-r, r-1] where r = rank(data) but Burn only supports positive dim
                    dim += tensor.rank as i64;
                }
                dim as usize
            })
            .collect();

        // Sort the dimensions to ensure consistent order
        dims.sort();

        let config = ReduceConfig::new(dims, keepdims == 1);
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        log::debug!("{} rank inference for node {}", node.node_type, node.name);

        // Extract tensor info before calling process_config
        let (tensor_rank, tensor_elem_type, tensor_static_shape) = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => (
                tensor.rank,
                tensor.elem_type.clone(),
                tensor.static_shape.clone(),
            ),
            _ => panic!("{}: Only tensor input is valid", node.node_type),
        };
        log::debug!(
            "{} input rank for {}: {}",
            node.node_type,
            node.name,
            tensor_rank
        );

        let config = node.config::<ReduceConfig>();

        log::debug!(
            "{} config for {}: keepdims={}, dims={:?}",
            node.node_type,
            node.name,
            config.keepdims,
            config.dims
        );
        log::debug!(
            "{} static_shape for {}: {:?}",
            node.node_type,
            node.name,
            tensor_static_shape
        );

        // Determine if the output should be a scalar
        let should_be_scalar =
            !config.keepdims && (config.dims.is_empty() || config.dims.len() == tensor_rank);

        if should_be_scalar {
            // Output is a scalar
            log::debug!("{} output is scalar for node {}", node.node_type, node.name);
            node.outputs[0].ty = ArgType::Scalar(tensor_elem_type);
        } else {
            // Output is a tensor
            let output_rank = if config.keepdims {
                tensor_rank
            } else {
                tensor_rank - config.dims.len()
            };

            // Infer static shape based if given
            let output_shape = tensor_static_shape.and_then(|mut shape| {
                log::debug!(
                    "{} processing static_shape for {}: shape.len()={}, dims={:?}",
                    node.node_type,
                    node.name,
                    shape.len(),
                    config.dims
                );

                // Only process static shape if it's complete (matches tensor rank)
                if shape.len() != tensor_rank {
                    log::debug!(
                        "{} skipping static_shape for {}: shape.len()={} != rank={}",
                        node.node_type,
                        node.name,
                        shape.len(),
                        tensor_rank
                    );
                    return None;
                }

                if config.keepdims {
                    for dim in &config.dims {
                        log::debug!(
                            "{} setting shape[{}] = 1 for {} (shape.len()={})",
                            node.node_type,
                            dim,
                            node.name,
                            shape.len()
                        );
                        shape[*dim] = 1;
                    }
                    Some(shape)
                } else {
                    for dim in config.dims.iter().rev() {
                        shape.remove(*dim);
                    }
                    Some(shape)
                }
            });

            log::debug!(
                "{} output rank for {}: {}",
                node.node_type,
                node.name,
                output_rank
            );

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: tensor_elem_type,
                rank: output_rank,
                static_shape: output_shape,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axes: Option<Vec<i64>>, keepdims: Option<i64>) -> Node {
        let mut builder = NodeBuilder::new(NodeType::ReduceMax, "test_reduce_max")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("reduced", 3, None);

        if let Some(axes_val) = axes {
            builder = builder.attr_ints("axes", axes_val);
        }
        if let Some(kd) = keepdims {
            builder = builder.attr_int("keepdims", kd);
        }

        builder.build()
    }

    #[test]
    fn test_reduce_config_basic() {
        let node = create_test_node(Some(vec![1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;

        processor.process_config(&mut node, 16);

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_negative_axis() {
        let node = create_test_node(Some(vec![-2]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;

        processor.process_config(&mut node, 16);

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]); // -2 + 3 = 1
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_axes() {
        let node = create_test_node(None, Some(1));
        let mut node = node;

        let processor = ReduceProcessor;

        processor.process_config(&mut node, 16);

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, []);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_multiple_axes() {
        let node = create_test_node(Some(vec![0, 1]), Some(1));
        let mut node = node;

        let processor = ReduceProcessor;

        processor.process_config(&mut node, 16);

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [0, 1]);
        assert_eq!(config.keepdims, true);
    }

    #[test]
    fn test_reduce_config_no_keepdims() {
        let node = create_test_node(Some(vec![1]), Some(0));
        let mut node = node;

        let processor = ReduceProcessor;

        processor.process_config(&mut node, 16);

        let config = node.config::<ReduceConfig>();

        assert_eq!(config.dims, [1]);
        assert_eq!(config.keepdims, false);
    }

    #[test]
    fn test_reduce_update_outputs_scalar_no_axes_no_keepdims() {
        // Test that reduce with no axes and keepdims=false produces a scalar output
        let mut node = create_test_node(None, Some(0));
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_scalar_all_dims_no_keepdims() {
        // Test that reduce with all dimensions and keepdims=false produces a scalar output
        let mut node = create_test_node(Some(vec![0, 1, 2]), Some(0));
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Scalar(_) => {
                // This is the expected case - scalar output
            }
            ArgType::Tensor(_) => {
                panic!("Expected scalar output but got tensor");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_partial_dims_no_keepdims() {
        // Test that reduce with partial dimensions and keepdims=false produces a tensor output
        let mut node = create_test_node(Some(vec![1]), Some(0));
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2)
                assert_eq!(tensor.rank, 2);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_tensor_with_keepdims() {
        // Test that reduce with keepdims=true always produces a tensor output
        let mut node = create_test_node(None, Some(1));
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain original rank when keepdims=true
                assert_eq!(tensor.rank, 3);
            }
            ArgType::Scalar(_) => {
                panic!("Expected tensor output but got scalar when keepdims=true");
            }
            _ => {
                panic!("Unexpected output type");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_keepdims() {
        // Regression test for partial static_shape with keepdims=true
        // This was causing "index out of bounds" panic before the fix
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_partial_static_shape_no_keepdims() {
        // Regression test for partial static_shape without keepdims
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![768])) // Rank 3 but only last dim known
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![1]) // Reduce on dimension 1
            .attr_int("keepdims", 0)
            .build();

        // This should not panic
        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should be rank 2 (3 - 1 = 2) without keepdims
                assert_eq!(tensor.rank, 2);
                // Static shape should be None since input shape was partial
                assert_eq!(tensor.static_shape, None);
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }

    #[test]
    fn test_reduce_update_outputs_complete_static_shape_keepdims() {
        // Test that complete static_shape is properly updated with keepdims=true
        let mut node = NodeBuilder::new(NodeType::ReduceMean, "test_reduce_mean")
            .input_tensor_f32("data", 3, Some(vec![2, 4, 768])) // Complete shape
            .output_tensor_f32("reduced", 3, None)
            .attr_ints("axes", vec![2]) // Reduce on dimension 2
            .attr_int("keepdims", 1)
            .build();

        let processor = ReduceProcessor;
        processor.process_config(&mut node, 16);
        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                // Should maintain rank 3 with keepdims=true
                assert_eq!(tensor.rank, 3);
                // Static shape should be updated: [2, 4, 768] -> [2, 4, 1]
                assert_eq!(tensor.static_shape, Some(vec![2, 4, 1]));
            }
            _ => {
                panic!("Expected tensor output");
            }
        }
    }
}
