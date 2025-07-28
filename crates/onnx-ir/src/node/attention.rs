use crate::{ArgType, Argument, Node, TensorType};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionQkMatmulOutputMode {
    Matmul,
    MatmulPlusAttentionMask,
    MatmulAfterSoftcap,
    MatmulAfterSoftmax,
}

pub fn attention_config(node: &Node) -> AttentionConfig {
    if node.inputs.len() < 3 {
        panic!("Attention must have at least 3 inputs")
    }
    if node.outputs.is_empty() {
        panic!("Attention must have at least 1 output")
    }

    let q = extract_tensor(node.inputs.first(), "Q").unwrap();
    let k = extract_tensor(node.inputs.get(1), "K").unwrap();
    let v = extract_tensor(node.inputs.get(2), "V").unwrap();
    let y = extract_tensor(node.outputs.first(), "Y").unwrap();
    if q.rank != k.rank || q.rank != v.rank || q.rank != y.rank {
        panic!("Attention: Q, K, V, Y parameters must have the same rank");
    }
    if q.rank != 3 && q.rank != 4 {
        panic!("Attention: Q, K, V, Y parameters must have rank 3 or 4");
    }

    if (node.inputs.len() >= 6) != (node.outputs.len() >= 3)
        || node.inputs.len() == 5
        || node.outputs.len() == 2
    {
        panic!(
            "Attention: past_key, past_value, present_key, present_value can only be used together"
        );
    }

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
                    v => panic!(
                        "Unexpected value for attribute qk_matmul_output_mode for Attention: {v}"
                    ),
                }
            }
            "scale" => scale = Some(value.clone().into_f32() as f64),
            "softcap" => softcap = value.clone().into_f32() as f64,
            "softmax_precision" => softmax_precision = Some(value.clone().into_i64() as usize),
            _ => panic!("Unexpected attribute for Attention: {key}"),
        }
    }

    if q.rank == 3 && (kv_num_heads.is_none() || q_num_heads.is_none()) {
        panic!(
            "Attention: if Q, K, V are rank 3 the kv_num_heads and q_num_heads attributes must be specified"
        )
    }

    AttentionConfig::new(
        is_causal,
        q_num_heads,
        kv_num_heads,
        qk_matmul_output_mode,
        scale,
        softcap,
        softmax_precision,
    )
}

pub fn attention_update_output(node: &mut Node) {
    let q = extract_tensor(node.inputs.first(), "Q").unwrap();

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank: q.rank,
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
}

fn extract_tensor<'a>(arg: Option<&'a Argument>, name: &str) -> Option<&'a TensorType> {
    match &arg?.ty {
        ArgType::Tensor(v) => Some(v),
        _ => panic!("Attention: {name} input must be a tensor"),
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
    #[should_panic]
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
        attention_config(&node);
    }

    #[test]
    fn test_softcap() {
        let node = create_simple_test_node(None, None, None, None, None, Some(2.0), None);
        let config = attention_config(&node);
        assert_eq!(config.softcap, 2.0);
    }

    #[test]
    fn test_custom_scale() {
        let node = create_simple_test_node(None, None, None, None, Some(2.0), None, None);
        let config = attention_config(&node);
        assert_eq!(config.scale, Some(2.0));
    }

    #[test]
    fn test_is_causal() {
        let node = create_simple_test_node(Some(1), None, None, None, None, None, None);
        let config = attention_config(&node);
        assert!(config.is_causal);
    }

    #[rstest]
    #[case(0, AttentionQkMatmulOutputMode::Matmul)]
    #[case(1, AttentionQkMatmulOutputMode::MatmulPlusAttentionMask)]
    #[case(2, AttentionQkMatmulOutputMode::MatmulAfterSoftcap)]
    #[case(3, AttentionQkMatmulOutputMode::MatmulAfterSoftmax)]
    fn test_qk_matmul_output(#[case] raw: i64, #[case] mode: AttentionQkMatmulOutputMode) {
        let node = create_simple_test_node(None, None, None, Some(raw), None, None, None);
        let config = attention_config(&node);
        assert_eq!(config.qk_matmul_output_mode, mode);
    }
}
