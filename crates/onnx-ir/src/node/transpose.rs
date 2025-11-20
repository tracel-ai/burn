//! # Transpose
//!
//! Transposes the input tensor by permuting its dimensions, similar to numpy.transpose.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Transpose.html>
//!
//! ## Type Constraints
//! - T: All tensor types (float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
//!   uint32, uint64, bool, complex64, complex128, bfloat16, string)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version supporting all tensor types and permutation.
//! - **Opset 13**: Added bfloat16 support.
//! - **Opset 21**: Added support for int4, uint4, and float8 types.
//!
//! ## Example
//! When `perm = [1, 0, 2]` and input shape is `(1, 2, 3)`, the output shape will be `(2, 1, 3)`.

use derive_new::new;
use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{ArgType, Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Configuration for Transpose operations
#[derive(Debug, Clone, new)]
pub struct TransposeConfig {
    /// Permutation of dimensions
    pub perm: Vec<i64>,
}

/// Node representation for Transpose operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct TransposeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: TransposeConfig,
}

pub(crate) struct TransposeProcessor;

impl NodeProcessor for TransposeProcessor {
    type Config = TransposeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
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

        // TODO: Missing validation that perm is a valid permutation.
        // Must verify: len(perm) == rank, all values in [0, rank-1], no duplicates.
        // Invalid permutations could cause index out of bounds errors.

        // TODO: Missing validation that perm contains each index exactly once.
        // E.g., perm=[0, 0, 2] has duplicate 0, missing 1 - should be rejected.

        // Validate perm length matches input rank
        let input_rank = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        if config.perm.len() != input_rank {
            return Err(ProcessError::Custom(format!(
                "Transpose: perm length {} doesn't match input rank {}",
                config.perm.len(),
                input_rank
            )));
        }

        // Infer output type
        same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Default: reverse the dimensions
        let mut perm = (0..tensor.rank as i64).rev().collect::<Vec<i64>>();

        if let Some(axes) = node.attrs.get("perm") {
            perm = axes.clone().into_i64s();

            // TODO: Validate perm values are in valid range [0, rank-1].
            // Out-of-bounds values in perm should be rejected early.
        }

        let config = TransposeConfig { perm };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Transpose(TransposeNode {
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

    fn create_test_node(perm: Option<Vec<i64>>, rank: usize) -> NodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Transpose, "test_transpose")
            .input_tensor_f32("data", rank, None)
            .output_tensor_f32("transposed", rank, None);

        if let Some(perm_val) = perm {
            builder = builder.attr_ints("perm", perm_val);
        }

        builder.build()
    }

    #[test]
    fn test_transpose_config_default() {
        let node = create_test_node(None, 3);
        let mut node = node;
        let processor = TransposeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.perm, vec![2, 1, 0]); // Default is to reverse the dimensions
    }

    #[test]
    fn test_transpose_config_with_perm() {
        let node = create_test_node(Some(vec![0, 2, 1]), 3);
        let mut node = node;
        let processor = TransposeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.perm, vec![0, 2, 1]);
    }

    #[test]
    fn test_transpose_config_multiple_inputs() {
        let mut node = create_test_node(None, 3);
        // Add an extra input to cause the expected error
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                dtype: crate::ir::DType::F32,
                rank: 3,
                static_shape: None,
            }),
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        let processor = TransposeProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    // TODO: Missing test for invalid permutations - duplicate indices.
    // E.g., perm=[0, 0, 2] should be rejected.

    // TODO: Missing test for out-of-bounds indices in perm.
    // E.g., perm=[0, 5, 2] for rank-3 tensor should be rejected.

    // TODO: Missing test for perm length mismatch.
    // E.g., perm=[0, 1] for rank-3 tensor should be rejected.

    // TODO: Missing test for 1D tensor transpose - perm=[0] should be identity.

    // TODO: Missing test for 5D+ tensors - verify works for higher ranks.

    // TODO: Missing test for negative values in perm.
    // ONNX spec doesn't support negative indices in perm, should be rejected.
}
