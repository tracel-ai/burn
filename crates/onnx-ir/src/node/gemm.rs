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
        crate::util::validate_opset(opset, 11)?;
        crate::util::validate_min_inputs(node, 2)?;
        crate::util::validate_output_count(node, 1)?;

        log::debug!("Gemm rank inference for node {}", node.name);

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

        log::debug!(
            "Gemm input ranks for {}: a_rank={}, b_rank={}",
            node.name,
            a_rank,
            b_rank
        );

        let output_rank = max(a_rank, b_rank);
        log::debug!("Gemm output rank for {}: {}", node.name, output_rank);

        let elem_type = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.elem_type.clone(),
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
            elem_type,
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
