use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{ArgType, Argument, Node, NodeConfig, TensorType};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub is_causal: bool,
    pub kv_num_heads: Option<usize>,
    pub q_num_heads: Option<usize>,
    pub qk_matmul_output_mode: AttentionQkMatmulOutputMode,
    pub scale: Option<f64>,
    pub softcap: f64,
    pub softmax_precision: Option<usize>,
}

impl AttentionConfig {
    pub fn new(
        is_causal: bool,
        kv_num_heads: Option<usize>,
        q_num_heads: Option<usize>,
        qk_matmul_output_mode: AttentionQkMatmulOutputMode,
        scale: Option<f64>,
        softcap: f64,
        softmax_precision: Option<usize>,
    ) -> Self {
        Self {
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode,
            scale,
            softcap,
            softmax_precision,
        }
    }
}

impl NodeConfig for AttentionConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionQkMatmulOutputMode {
    Matmul,
    MatmulPlusAttentionMask,
    MatmulAfterSoftcap,
    MatmulAfterSoftmax,
}

fn extract_tensor<'a>(
    arg: Option<&'a Argument>,
    name: &str,
) -> Result<Option<&'a TensorType>, ProcessError> {
    match arg {
        None => Ok(None),
        Some(a) => match &a.ty {
            ArgType::Tensor(v) => Ok(Some(v)),
            _ => Err(ProcessError::Custom(format!(
                "Attention: {name} input must be a tensor"
            ))),
        },
    }
}

pub struct AttentionProcessor;

impl NodeProcessor for AttentionProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        const MIN: usize = 23;

        // Attention implementation supports opset 23+
        if opset < MIN {
            return Err(ProcessError::UnsupportedOpset {
                required: MIN,
                actual: opset,
            });
        }

        if node.inputs.len() < 3 {
            return Err(ProcessError::InvalidInputCount {
                expected: 3,
                actual: node.inputs.len(),
            });
        }
        if node.outputs.is_empty() {
            return Err(ProcessError::InvalidOutputCount {
                expected: 1,
                actual: node.outputs.len(),
            });
        }

        let q = extract_tensor(node.inputs.first(), "Q")?.ok_or_else(|| {
            ProcessError::Custom("Attention: Q input must be present".to_string())
        })?;
        let k = extract_tensor(node.inputs.get(1), "K")?.ok_or_else(|| {
            ProcessError::Custom("Attention: K input must be present".to_string())
        })?;
        let v = extract_tensor(node.inputs.get(2), "V")?.ok_or_else(|| {
            ProcessError::Custom("Attention: V input must be present".to_string())
        })?;

        // Validate that Q, K, V have the same rank (output Y will be inferred)
        if q.rank != k.rank || q.rank != v.rank {
            return Err(ProcessError::Custom(
                "Attention: Q, K, V parameters must have the same rank".to_string(),
            ));
        }
        if q.rank != 3 && q.rank != 4 {
            return Err(ProcessError::Custom(
                "Attention: Q, K, V, Y parameters must have rank 3 or 4".to_string(),
            ));
        }

        if (node.inputs.len() >= 6) != (node.outputs.len() >= 3)
            || node.inputs.len() == 5
            || node.outputs.len() == 2
        {
            return Err(ProcessError::Custom(
                "Attention: past_key, past_value, present_key, present_value can only be used together".to_string(),
            ));
        }

        // Validate unexpected attributes and qk_matmul_output_mode before config extraction
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "is_causal" | "kv_num_heads" | "q_num_heads" | "scale" | "softcap"
                | "softmax_precision" => {}
                "qk_matmul_output_mode" => match value.clone().into_i64() {
                    0..=3 => {}
                    v => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "qk_matmul_output_mode".to_string(),
                            reason: format!(
                                "Unexpected value for attribute qk_matmul_output_mode for Attention: {v}"
                            ),
                        });
                    }
                },
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Attention: {key}"),
                    });
                }
            }
        }

        // Get reference to config for validation
        let config = node.config::<AttentionConfig>();

        if q.rank == 3 && (config.kv_num_heads.is_none() || config.q_num_heads.is_none()) {
            return Err(ProcessError::Custom(
                "Attention: if Q, K, V are rank 3 the kv_num_heads and q_num_heads attributes must be specified".to_string(),
            ));
        }

        // Infer output types
        let q_rank = q.rank;
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: node.inputs[0].ty.elem_type().clone(),
            rank: q_rank,
            static_shape: None,
        });

        if let Some(present_key) = node.outputs.get_mut(1) {
            present_key.ty = ArgType::Tensor(TensorType {
                elem_type: node.inputs[4].ty.elem_type().clone(),
                rank: 4,
                static_shape: None,
            });
        }

        if let Some(present_value) = node.outputs.get_mut(2) {
            present_value.ty = ArgType::Tensor(TensorType {
                elem_type: node.inputs[5].ty.elem_type().clone(),
                rank: 4,
                static_shape: None,
            });
        }

        if let Some(qk_matmul_output) = node.outputs.get_mut(3) {
            qk_matmul_output.ty = ArgType::Tensor(TensorType {
                elem_type: node.inputs[0].ty.elem_type().clone(),
                rank: 4,
                static_shape: None,
            });
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let _q = extract_tensor(node.inputs.first(), "Q")?.ok_or_else(|| {
            ProcessError::Custom("Attention: Q input must be present".to_string())
        })?;

        let mut is_causal = false;
        let mut kv_num_heads = None;
        let mut q_num_heads = None;
        let mut qk_matmul_output_mode = AttentionQkMatmulOutputMode::Matmul;
        let mut scale = None;
        let mut softcap = 0.0;
        let mut softmax_precision = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "is_causal" => is_causal = value.clone().into_i64() != 0,
                "kv_num_heads" => kv_num_heads = Some(value.clone().into_i64() as usize),
                "q_num_heads" => q_num_heads = Some(value.clone().into_i64() as usize),
                "qk_matmul_output_mode" => {
                    qk_matmul_output_mode = match value.clone().into_i64() {
                        0 => AttentionQkMatmulOutputMode::Matmul,
                        1 => AttentionQkMatmulOutputMode::MatmulPlusAttentionMask,
                        2 => AttentionQkMatmulOutputMode::MatmulAfterSoftcap,
                        3 => AttentionQkMatmulOutputMode::MatmulAfterSoftmax,
                        _ => AttentionQkMatmulOutputMode::Matmul, // Use default for unknown values
                    }
                }
                "scale" => scale = Some(value.clone().into_f32() as f64),
                "softcap" => softcap = value.clone().into_f32() as f64,
                "softmax_precision" => softmax_precision = Some(value.clone().into_i64() as usize),
                _ => {}
            }
        }

        let config = AttentionConfig::new(
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode,
            scale,
            softcap,
            softmax_precision,
        );
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
mod tests {
    use super::*;
    use crate::{ElementType, NodeType, node::test_utils::NodeBuilder};
    use rstest::rstest;

    fn create_test_node(
        q: Option<usize>,
        k: Option<usize>,
        v: Option<usize>,
        attn_mask: Option<(ElementType, usize)>,
        past_key: Option<usize>,
        past_value: Option<usize>,
        y: Option<usize>,
        present_key: Option<usize>,
        present_value: Option<usize>,
        qk_matmul_output: Option<usize>,
        is_causal: Option<i64>,
        kv_num_heads: Option<i64>,
        q_num_heads: Option<i64>,
        qk_matmul_output_mode: Option<i64>,
        scale: Option<f32>,
        softcap: Option<f32>,
        softmax_precision: Option<i64>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Attention, "test_attention");

        if let Some(rank) = q {
            builder = builder.input_tensor_f32("q", rank, None);
        }
        if let Some(rank) = k {
            builder = builder.input_tensor_f32("k", rank, None);
        }
        if let Some(rank) = v {
            builder = builder.input_tensor_f32("v", rank, None);
        }
        if let Some((ty, rank)) = attn_mask {
            builder = builder.add_input(
                "attn_mask",
                ArgType::Tensor(TensorType {
                    elem_type: ty,
                    rank,
                    static_shape: None,
                }),
            );
        }
        if let Some(rank) = past_key {
            builder = builder.input_tensor_f32("past_key", rank, None);
        }
        if let Some(rank) = past_value {
            builder = builder.input_tensor_f32("past_value", rank, None);
        }
        if let Some(rank) = y {
            builder = builder.output_tensor_f32("y", rank, None);
        }
        if let Some(rank) = present_key {
            builder = builder.output_tensor_f32("present_key", rank, None);
        }
        if let Some(rank) = present_value {
            builder = builder.output_tensor_f32("present_value", rank, None);
        }
        if let Some(rank) = qk_matmul_output {
            builder = builder.output_tensor_f32("qk_matmul_output", rank, None);
        }

        if let Some(is_causal) = is_causal {
            builder = builder.attr_int("is_causal", is_causal);
        }
        if let Some(kv_num_heads) = kv_num_heads {
            builder = builder.attr_int("kv_num_heads", kv_num_heads);
        }
        if let Some(q_num_heads) = q_num_heads {
            builder = builder.attr_int("q_num_heads", q_num_heads);
        }
        if let Some(qk_matmul_output_mode) = qk_matmul_output_mode {
            builder = builder.attr_int("qk_matmul_output_mode", qk_matmul_output_mode);
        }
        if let Some(scale) = scale {
            builder = builder.attr_float("scale", scale);
        }
        if let Some(softcap) = softcap {
            builder = builder.attr_float("softcap", softcap);
        }
        if let Some(softmax_precision) = softmax_precision {
            builder = builder.attr_int("softmax_precision", softmax_precision);
        }

        builder.build()
    }

    fn create_simple_test_node(
        is_causal: Option<i64>,
        kv_num_heads: Option<i64>,
        q_num_heads: Option<i64>,
        qk_matmul_output_mode: Option<i64>,
        scale: Option<f32>,
        softcap: Option<f32>,
        softmax_precision: Option<i64>,
    ) -> Node {
        create_test_node(
            Some(4),
            Some(4),
            Some(4),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
            is_causal,
            kv_num_heads,
            q_num_heads,
            qk_matmul_output_mode,
            scale,
            softcap,
            softmax_precision,
        )
    }

    #[rstest]
    // Missing required inputs or outputs
    #[case(None, Some(4), Some(4), None, None, None, Some(4), None, None)]
    #[case(Some(4), None, Some(4), None, None, None, Some(4), None, None)]
    #[case(Some(4), Some(4), None, None, None, None, Some(4), None, None)]
    #[case(Some(4), Some(4), Some(4), None, None, None, None, None, None)]
    #[case(Some(4), Some(4), None, None, None, None, None, None, None)]
    #[case(Some(4), Some(4), Some(4), Some((ElementType::Bool,2)), Some(2), None, Some(4), None, None)]
    #[case(Some(4), Some(4), Some(4), None, None, None, Some(4), Some(2), None)]
    #[case(Some(4), Some(4), Some(4), Some((ElementType::Bool,2)), Some(2), Some(2), Some(4), None, None)]
    // Mismatched ranks
    #[case(Some(4), Some(3), Some(3), None, None, None, Some(3), None, None)]
    #[case(Some(3), Some(4), Some(3), None, None, None, Some(4), None, None)]
    #[case(Some(3), Some(3), Some(4), None, None, None, Some(1), None, None)]
    // 3D qkv inputs without the *_num_heads attributes
    #[case(Some(3), Some(3), Some(3), None, None, None, Some(3), None, None)]
    fn test_fail_on_invalid_inputs(
        #[case] q: Option<usize>,
        #[case] k: Option<usize>,
        #[case] v: Option<usize>,
        #[case] attn_mask: Option<(ElementType, usize)>,
        #[case] past_key: Option<usize>,
        #[case] past_value: Option<usize>,
        #[case] y: Option<usize>,
        #[case] present_key: Option<usize>,
        #[case] present_value: Option<usize>,
    ) {
        let node = create_test_node(
            q,
            k,
            v,
            attn_mask,
            past_key,
            past_value,
            y,
            present_key,
            present_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut node = node;
        let processor = AttentionProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 23, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_softcap() {
        let node = create_simple_test_node(None, None, None, None, None, Some(2.0), None);
        let mut node = node;
        let processor = AttentionProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 23, &prefs).unwrap();
        let config = node.config::<AttentionConfig>();
        assert_eq!(config.softcap, 2.0);
    }

    #[test]
    fn test_custom_scale() {
        let node = create_simple_test_node(None, None, None, None, Some(2.0), None, None);
        let mut node = node;
        let processor = AttentionProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 23, &prefs).unwrap();
        let config = node.config::<AttentionConfig>();
        assert_eq!(config.scale, Some(2.0));
    }

    #[test]
    fn test_is_causal() {
        let node = create_simple_test_node(Some(1), None, None, None, None, None, None);
        let mut node = node;
        let processor = AttentionProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 23, &prefs).unwrap();
        let config = node.config::<AttentionConfig>();
        assert!(config.is_causal);
    }

    #[rstest]
    #[case(0, AttentionQkMatmulOutputMode::Matmul)]
    #[case(1, AttentionQkMatmulOutputMode::MatmulPlusAttentionMask)]
    #[case(2, AttentionQkMatmulOutputMode::MatmulAfterSoftcap)]
    #[case(3, AttentionQkMatmulOutputMode::MatmulAfterSoftmax)]
    fn test_qk_matmul_output(#[case] raw: i64, #[case] mode: AttentionQkMatmulOutputMode) {
        let node = create_simple_test_node(None, None, None, Some(raw), None, None, None);
        let mut node = node;
        let processor = AttentionProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 23).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 23, &prefs).unwrap();
        let config = node.config::<AttentionConfig>();
        assert_eq!(config.qk_matmul_output_mode, mode);
    }
}
