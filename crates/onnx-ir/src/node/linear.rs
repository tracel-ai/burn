use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::NodeProcessor;
use std::any::Any;

/// Configuration for Linear operations
#[derive(Debug, Clone)]
pub struct LinearConfig {
    /// Input dimension (features)
    pub d_input: usize,
    /// Output dimension (features)
    pub d_output: usize,
    /// Whether bias is used
    pub bias: bool,
}

impl LinearConfig {
    /// Create a new LinearConfig
    pub fn new(d_input: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_output,
            bias: true,
        }
    }

    /// Set whether bias is used
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

impl NodeConfig for LinearConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct LinearProcessor;

impl NodeProcessor for LinearProcessor {
    fn process_config(&self, node: &mut Node, _opset: usize) {
        if node.inputs.len() < 2 {
            panic!("Linear: missing weight tensor");
        }

        let weight_shape = node.inputs[1]
            .into_value()
            .expect("Linear: weight tensor must be present")
            .shape
            .clone();

        // check if the weight tensor has at least 2 dimensions
        if weight_shape.len() < 2 {
            panic!(
                "Linear: weight tensor must have at least 2 dimensions (got {:?})",
                weight_shape.len()
            );
        }

        let (in_size, out_size) = (weight_shape[0], weight_shape[1]);

        // check if the bias is present
        let bias = node.inputs.len() == 3 && node.inputs[2].into_value().is_some();

        let config = LinearConfig::new(in_size, out_size).with_bias(bias);
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        log::debug!("Linear rank inference for node {}", node.name);

        if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
            log::debug!("Linear input rank for {}: {}", node.name, tensor.rank);

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: tensor.elem_type.clone(),
                rank: tensor.rank,
                static_shape: None,
            });

            log::debug!("Linear output rank for {}: {}", node.name, tensor.rank);
        } else {
            panic!("Only tensor input is valid");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(has_bias: bool, weight_dims: Vec<usize>) -> NodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.0; weight_dims.iter().product()]; // Not important for the test

        // Start building the node with input and weight
        let mut builder = NodeBuilder::new(NodeType::Gemm, "test_linear")
            .input_tensor_f32("input", 2, None)
            .input_tensor_f32_data("weight", weight_data, weight_dims.clone())
            .output_tensor_f32("output", 2, None);

        // Add bias if needed
        if has_bias {
            let bias_data = vec![0.0; weight_dims[1]]; // bias size equals output size
            builder = builder.input_tensor_f32_data("bias", bias_data, vec![weight_dims[1]]);
        }

        builder
    }

    #[test]
    fn test_linear_config_basic() {
        let node = create_test_node(false, vec![10, 5]).build_with_graph_data(16);
        let mut node = node;
        let processor = LinearProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<LinearConfig>();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(!config.bias);
    }

    #[test]
    fn test_linear_config_with_bias() {
        let node = create_test_node(true, vec![10, 5]).build_with_graph_data(16);
        let mut node = node;
        let processor = LinearProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<LinearConfig>();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Linear: weight tensor must have at least 2 dimensions")]
    fn test_linear_config_invalid_weight_dims() {
        let node = create_test_node(false, vec![10]).build_with_graph_data(16);
        let mut node = node;
        let processor = LinearProcessor;
        processor.process_config(&mut node, 16);
    }

    #[test]
    #[should_panic(expected = "Linear: missing weight tensor")]
    fn test_linear_config_missing_weight() {
        let mut node = create_test_node(false, vec![10, 5]).build_with_graph_data(16);
        node.inputs.remove(1);
        let mut node = node;
        let processor = LinearProcessor;
        processor.process_config(&mut node, 16);
    }
}
