//! # Linear
//!
//! Linear transformation: Y = X * W^T + b
//!
//! **Note**: This is a Burn-specific node type created by fusing ONNX Gemm or MatMul+Add operations.
//! See the node_conversion phase where Gemm (with alpha=1, beta=1, transB=1) is converted to Linear,
//! and MatMul followed by Add is fused into Linear.
//!
//! **Related ONNX Specs**:
//! - Gemm: <https://onnx.ai/onnx/operators/onnx__Gemm.html>
//! - MatMul: <https://onnx.ai/onnx/operators/onnx__MatMul.html>
//!
//! ## Attributes (None - converted from source operators)
//!
//! ## Inputs
//! - `X` (T): Input tensor
//! - `W` (T): Weight matrix (transposed during conversion)
//! - `b` (T, optional): Bias vector
//!
//! ## Outputs
//! - `Y` (T): Output tensor

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
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
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift weight (input 1) and bias (input 2) if present
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
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_min_inputs(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Validate that no unexpected attributes are present
        // Linear is a Burn-specific node type with no attributes
        if let Some((key, _value)) = node.attrs.iter().next() {
            return Err(ProcessError::InvalidAttribute {
                name: key.clone(),
                reason: format!("Linear does not accept any attributes, found: {}", key),
            });
        }

        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: tensor.dtype,
            rank: tensor.rank,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Linear: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        let (in_size, out_size) = (weight_shape[0], weight_shape[1]);

        // check if the bias is present (could be Constant, Static, or Dynamic)
        let bias = node.inputs.len() == 3 && !node.inputs[2].is_optional();

        let config = LinearConfig::new(in_size, out_size).with_bias(bias);
        Ok(Some(Box::new(config)))
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
        let node = create_test_node(false, vec![10, 5]).process(LinearProcessor, 16);
        let config = node.config::<LinearConfig>();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(!config.bias);
    }

    #[test]
    fn test_linear_config_with_bias() {
        let node = create_test_node(true, vec![10, 5]).process(LinearProcessor, 16);
        let config = node.config::<LinearConfig>();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_linear_config_invalid_weight_dims() {
        let _node = create_test_node(false, vec![10]).process(LinearProcessor, 16);
        // Error should occur in extract_config when accessing weight_shape[1]
    }

    #[test]
    fn test_linear_config_missing_weight() {
        let mut node = create_test_node(false, vec![10, 5]).build_with_graph_data(16);
        node.inputs.remove(1);

        let processor = LinearProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }
}
