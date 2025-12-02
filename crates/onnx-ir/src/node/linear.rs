//! # Linear
//!
//! Linear transformation: Y = X * W^T + b
//!
//! **Note**: This is a Burn-specific node type created by fusing ONNX Gemm or MatMul+Add operations.
//! See the node_conversion phase where Gemm (with alpha=1, beta=1, transB=1) is converted to Linear,
//! and MatMul followed by Add is fused into Linear.
//!
//! **Weight Layout**: The weight tensor layout depends on the source operation:
//! - **Gemm-sourced** (transB=1): Weight is in `[out_features, in_features]` format.
//!   The `transpose_weight` config flag is set to `true`.
//! - **MatMul-sourced**: Weight is already in `[in_features, out_features]` format.
//!   The `transpose_weight` config flag is set to `false`.
//!
//! The burn-import layer reads the `transpose_weight` flag and transposes the weight
//! tensor only when needed during code generation.
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

use derive_new::new;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for Linear operations
#[derive(Debug, Clone, new)]
pub struct LinearConfig {
    /// Input dimension (features)
    pub d_input: usize,
    /// Output dimension (features)
    pub d_output: usize,
    /// Whether bias is used
    pub bias: bool,
    /// Whether weight needs transposition for Burn's expected layout.
    /// - true: Weight is in ONNX Gemm layout [out_features, in_features] (from Gemm with transB=1)
    /// - false: Weight is already in [in_features, out_features] format (from MatMul)
    pub transpose_weight: bool,
}

/// Node representation for Linear operation
#[derive(Debug, Clone)]
pub struct LinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LinearConfig,
}

pub(crate) struct LinearProcessor;

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

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
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
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate weight tensor (input 1) is exactly 2D - Higher or lower rank weights are invalid - burn/crates/onnx-ir/src/node/linear.rs:86
        // TODO: Validate all inputs have compatible dtypes - Type mismatch would cause runtime errors - burn/crates/onnx-ir/src/node/linear.rs:86
        // TODO: Validate input rank is compatible for matrix multiplication - At least 2D required - burn/crates/onnx-ir/src/node/linear.rs:86

        // Validate that only expected attributes are present
        // Linear accepts only transpose_weight (internal attribute set by node_conversion)
        for key in node.attrs.keys() {
            if key != "transpose_weight" {
                return Err(ProcessError::InvalidAttribute {
                    name: key.clone(),
                    reason: format!(
                        "Linear only accepts 'transpose_weight' attribute, found: {}",
                        key
                    ),
                });
            }
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

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        use crate::ir::AttributeValue;

        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("Linear: weight tensor must be present".to_string())
            })?
            .shape
            .to_vec();

        // Check if weight needs transposition (set by node_conversion phase)
        // - Gemm with transB=1 → transpose_weight=true (weight is [out, in])
        // - MatMul → transpose_weight=false (weight is [in, out])
        let transpose_weight = node
            .attrs
            .get("transpose_weight")
            .map(|v| matches!(v, AttributeValue::Int64(1)))
            .unwrap_or(false);

        // TODO: Validate weight_shape.len() == 2 - Linear requires exactly 2D weight matrix
        let (in_size, out_size) = if transpose_weight {
            // Gemm layout: [out_features, in_features]
            (weight_shape[1], weight_shape[0])
        } else {
            // MatMul layout: [in_features, out_features]
            (weight_shape[0], weight_shape[1])
        };

        // check if the bias is present (could be Constant, Static, or Dynamic)
        let bias = node.inputs.len() == 3 && !node.inputs[2].is_optional();

        let config = LinearConfig::new(in_size, out_size, bias, transpose_weight);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Linear(LinearNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    /// Create a test node simulating Gemm->Linear conversion
    /// Weight is in ONNX Gemm layout [out_features, in_features] with transpose_weight=true
    fn create_gemm_linear_node(has_bias: bool, weight_dims: Vec<usize>) -> TestNodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.0; weight_dims.iter().product()]; // Not important for the test

        // Start building the node with input and weight
        // weight_dims is in Gemm format: [out_features, in_features]
        let mut builder = TestNodeBuilder::new(NodeType::Linear, "test_linear")
            .input_tensor_f32("input", 2, None)
            .input_tensor_f32_data("weight", weight_data, weight_dims.clone())
            .output_tensor_f32("output", 2, None)
            .attr_int("transpose_weight", 1); // Gemm-sourced

        // Add bias if needed - bias size equals out_features (weight_dims[0])
        if has_bias {
            let bias_data = vec![0.0; weight_dims[0]];
            builder = builder.input_tensor_f32_data("bias", bias_data, vec![weight_dims[0]]);
        }

        builder
    }

    /// Create a test node simulating MatMul->Linear conversion
    /// Weight is in MatMul layout [in_features, out_features] with transpose_weight=false
    fn create_matmul_linear_node(has_bias: bool, weight_dims: Vec<usize>) -> TestNodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.0; weight_dims.iter().product()]; // Not important for the test

        // Start building the node with input and weight
        // weight_dims is in MatMul format: [in_features, out_features]
        let mut builder = TestNodeBuilder::new(NodeType::Linear, "test_linear")
            .input_tensor_f32("input", 2, None)
            .input_tensor_f32_data("weight", weight_data, weight_dims.clone())
            .output_tensor_f32("output", 2, None);
        // No transpose_weight attribute means MatMul-sourced (transpose_weight=false)

        // Add bias if needed - bias size equals out_features (weight_dims[1] for MatMul)
        if has_bias {
            let bias_data = vec![0.0; weight_dims[1]];
            builder = builder.input_tensor_f32_data("bias", bias_data, vec![weight_dims[1]]);
        }

        builder
    }

    #[test]
    fn test_linear_config_gemm_source() {
        // Gemm layout: weight shape [10, 5] means [out_features=10, in_features=5]
        let node = create_gemm_linear_node(false, vec![10, 5]).process(LinearProcessor, 16);
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 5);
        assert_eq!(config.d_output, 10);
        assert!(!config.bias);
        assert!(config.transpose_weight);
    }

    #[test]
    fn test_linear_config_gemm_source_with_bias() {
        // Gemm layout: weight shape [10, 5] means [out_features=10, in_features=5]
        let node = create_gemm_linear_node(true, vec![10, 5]).process(LinearProcessor, 16);
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 5);
        assert_eq!(config.d_output, 10);
        assert!(config.bias);
        assert!(config.transpose_weight);
    }

    #[test]
    fn test_linear_config_matmul_source() {
        // MatMul layout: weight shape [5, 10] means [in_features=5, out_features=10]
        let node = create_matmul_linear_node(false, vec![5, 10]).process(LinearProcessor, 16);
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 5);
        assert_eq!(config.d_output, 10);
        assert!(!config.bias);
        assert!(!config.transpose_weight);
    }

    #[test]
    fn test_linear_config_matmul_source_with_bias() {
        // MatMul layout: weight shape [5, 10] means [in_features=5, out_features=10]
        let node = create_matmul_linear_node(true, vec![5, 10]).process(LinearProcessor, 16);
        let processor = LinearProcessor;
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.d_input, 5);
        assert_eq!(config.d_output, 10);
        assert!(config.bias);
        assert!(!config.transpose_weight);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_linear_config_invalid_weight_dims() {
        let node = create_matmul_linear_node(false, vec![10]).build_with_graph_data(16);
        let processor = LinearProcessor;
        // This should panic when accessing weight_shape[1] on a 1D weight tensor
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_linear_config_missing_weight() {
        let mut node = create_matmul_linear_node(false, vec![10, 5]).build_with_graph_data(16);
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
