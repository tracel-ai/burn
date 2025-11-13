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
//! ## Missing Test Coverage
//! - TODO: No test for Linear without bias (2 inputs only) - Optional bias not tested
//! - TODO: No test validating weight tensor must be 2D - 1D or 3D+ weights should be rejected
//! - TODO: No test for input rank validation - Spec requires specific input dimensions for matrix multiplication
//! - TODO: No test for dtype mismatch between inputs - All inputs should have same dtype
//! - TODO: No test for zero-size dimensions - Edge case for empty matrices
//! - TODO: Test uses sum verification instead of exact values - Could miss subtle bugs in weight application

use crate::ir::{ArgType, Node, NodeBuilder, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Linear operations
#[derive(Debug, Clone, Default)]
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

pub struct LinearProcessor;

impl NodeProcessor for LinearProcessor {
    type Config = LinearConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate weight tensor (input 1) is exactly 2D - Higher or lower rank weights are invalid - burn/crates/onnx-ir/src/node/linear.rs:86
        // TODO: Validate all inputs have compatible dtypes - Type mismatch would cause runtime errors - burn/crates/onnx-ir/src/node/linear.rs:86
        // TODO: Validate input rank is compatible for matrix multiplication - At least 2D required - burn/crates/onnx-ir/src/node/linear.rs:86

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
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Linear: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        // TODO: Validate weight_shape.len() == 2 - Linear requires exactly 2D weight matrix - burn/crates/onnx-ir/src/node/linear.rs:122
        let (in_size, out_size) = (weight_shape[0], weight_shape[1]);

        // check if the bias is present (could be Constant, Static, or Dynamic)
        let bias = node.inputs.len() == 3 && !node.inputs[2].is_optional();

        let config = LinearConfig::new(in_size, out_size).with_bias(bias);
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Linear {
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

    fn create_test_node(has_bias: bool, weight_dims: Vec<usize>) -> TestNodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.0; weight_dims.iter().product()]; // Not important for the test

        // Start building the node with input and weight
        let mut builder = TestNodeBuilder::new(NodeType::Gemm, "test_linear")
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
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(!config.bias);
    }

    #[test]
    fn test_linear_config_with_bias() {
        let node = create_test_node(true, vec![10, 5]).process(LinearProcessor, 16);
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_linear_config_invalid_weight_dims() {
        let node = create_test_node(false, vec![10]).build_with_graph_data(16);
        let processor = LinearProcessor;
        // This should panic when accessing weight_shape[1] on a 1D weight tensor
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_linear_config_missing_weight() {
        let mut node = create_test_node(false, vec![10, 5]).build_with_graph_data(16);
        node.inputs.remove(1);

        let processor = LinearProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }
}
