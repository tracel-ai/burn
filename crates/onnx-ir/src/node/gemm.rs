//! # Gemm
//!
//! General Matrix Multiplication: Y = alpha * A' * B' + beta * C
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Gemm.html>
//!
//! ## Description
//! Computes Y = alpha * A' * B' + beta * C, where:
//! - Input tensor A has shape (M, K) or (K, M) if transposed
//! - Input tensor B has shape (K, N) or (N, K) if transposed
//! - Input tensor C is broadcastable to shape (M, N)
//! - Output tensor Y has shape (M, N)
//!
//! ## Attributes
//! - `alpha` (float, default=1.0): Scalar multiplier for the product of input tensors A * B
//! - `beta` (float, default=1.0): Scalar multiplier for input tensor C
//! - `transA` (int, default=0): Whether to transpose A before computation (non-zero = transpose)
//! - `transB` (int, default=0): Whether to transpose B before computation (non-zero = transpose)
//!
//! ## Inputs
//! - `A` (T): Input tensor A. Shape is (M, K) if transA=0, or (K, M) if transA is non-zero
//! - `B` (T): Input tensor B. Shape is (K, N) if transB=0, or (N, K) if transB is non-zero
//! - `C` (T, optional): Optional input tensor C. If not specified, computation uses scalar 0. Shape should be unidirectional broadcastable to (M, N)
//!
//! ## Outputs
//! - `Y` (T): Output tensor of shape (M, N)
//!
//! ## Type Constraints
//! - T: tensor(float), tensor(double), tensor(float16), tensor(bfloat16), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic GEMM operation.
//! - **Opset 6**: Clarified broadcasting behavior for input C.
//! - **Opset 7**: Added support for additional data types (int32, int64, uint32, uint64).
//! - **Opset 11**: Changed attribute types to more specific types; clarified unidirectional broadcasting for C.
//! - **Opset 13**: Added bfloat16 and float16 support; updated type constraints.
//!
//! **Implementation Note**: This implementation validates opset 11+.
//!
//! ## Notes
//! In the node conversion phase, Gemm nodes are converted to Linear nodes when:
//! - `alpha` = 1.0
//! - `beta` = 1.0
//! - `transB` = 1
//!
//! This optimization allows the use of optimized Linear layer implementations in Burn.

use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use core::cmp::max;
use std::any::Any;

/// Configuration for Gemm operation
#[derive(Debug, Clone)]
pub struct GemmConfig {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: i64,
    pub trans_b: i64,
}

impl NodeConfig for GemmConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct GemmProcessor;

impl NodeProcessor for GemmProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 11)?;
        crate::processor::validate_min_inputs(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // Extract input A tensor type
        let a_rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Extract input B tensor type
        let b_rank = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        let output_rank = max(a_rank, b_rank);

        let elem_type = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.dtype,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: output_rank,
            static_shape: None,
            dtype: elem_type,
        });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let mut alpha: f32 = 1.0;
        let mut beta: f32 = 1.0;
        let mut trans_a: i64 = 0;
        let mut trans_b: i64 = 0;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => alpha = value.clone().into_f32(),
                "beta" => beta = value.clone().into_f32(),
                "transA" => trans_a = value.clone().into_i64(),
                "transB" => trans_b = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Gemm: {}", key),
                    });
                }
            }
        }

        let config = GemmConfig {
            alpha,
            beta,
            trans_a,
            trans_b,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i64>,
        trans_b: Option<i64>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Gemm, "test_gemm")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .input_tensor_f32("C", 2, None)
            .output_tensor_f32("Y", 2, None);

        if let Some(alpha_val) = alpha {
            builder = builder.attr_float("alpha", alpha_val);
        }
        if let Some(beta_val) = beta {
            builder = builder.attr_float("beta", beta_val);
        }
        if let Some(trans_a_val) = trans_a {
            builder = builder.attr_int("transA", trans_a_val);
        }
        if let Some(trans_b_val) = trans_b {
            builder = builder.attr_int("transB", trans_b_val);
        }

        builder.build()
    }

    #[test]
    fn test_gemm_config_defaults() {
        let node = create_test_node(None, None, None, None);
        let mut node = node;
        let processor = GemmProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GemmConfig>();
        assert_eq!(config.alpha, 1.0);
        assert_eq!(config.beta, 1.0);
        assert_eq!(config.trans_a, 0);
        assert_eq!(config.trans_b, 0);
    }

    #[test]
    fn test_gemm_config_with_attrs() {
        let node = create_test_node(Some(2.0), Some(3.0), Some(1), Some(1));
        let mut node = node;
        let processor = GemmProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GemmConfig>();
        assert_eq!(config.alpha, 2.0);
        assert_eq!(config.beta, 3.0);
        assert_eq!(config.trans_a, 1);
        assert_eq!(config.trans_b, 1);
    }

    #[test]
    fn test_gemm_config_partial_attrs() {
        let node = create_test_node(Some(0.5), None, Some(1), None);
        let mut node = node;
        let processor = GemmProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<GemmConfig>();
        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.beta, 1.0); // default
        assert_eq!(config.trans_a, 1);
        assert_eq!(config.trans_b, 0); // default
    }
}
