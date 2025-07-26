use crate::{ArgType, Argument, ElementType, Node, TensorType};

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub is_causal: bool,
    pub qk_matmul_output_mode: AttentionQkMatmulOutputMode,
    pub scale: f64,
    pub softcap: f64,
    pub softmax_precision: Option<usize>,
    pub computed_dims: AttentionDims,
}

impl AttentionConfig {
    pub fn new(
        is_causal: bool,
        qk_matmul_output_mode: AttentionQkMatmulOutputMode,
        scale: f64,
        softcap: f64,
        softmax_precision: Option<usize>,
        computed_dims: AttentionDims,
    ) -> Self {
        Self {
            is_causal,
            qk_matmul_output_mode,
            scale,
            softcap,
            softmax_precision,
            computed_dims,
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
    let dims = compute_dims(node);

    let mut is_causal = false;
    let mut qk_matmul_output_mode = AttentionQkMatmulOutputMode::Matmul;
    let mut maybe_scale = None;
    let mut softcap = 0.0;
    let mut softmax_precision = None;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "is_causal" => is_causal = value.clone().into_i64() != 0,
            "kv_num_heads" => (),
            "q_num_heads" => (),
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
            "scale" => maybe_scale = Some(value.clone().into_f32() as f64),
            "softcap" => softcap = value.clone().into_f32() as f64,
            "softmax_precision" => softmax_precision = Some(value.clone().into_i64() as usize),
            _ => panic!("Unexpected attribute for Attention: {key}"),
        }
    }

    let scale = maybe_scale.unwrap_or(1.0 / (dims.head_size as f64).sqrt());

    AttentionConfig::new(
        is_causal,
        qk_matmul_output_mode,
        scale,
        softcap,
        softmax_precision,
        dims,
    )
}

pub fn attention_update_output(node: &mut Node) {
    let dims = compute_dims(node);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank: dims.rank,
        static_shape: Some(match dims.rank {
            3 => vec![
                dims.batch_size,
                dims.q_sequence_length,
                dims.q_num_heads * dims.v_head_size,
            ],
            4 => vec![
                dims.batch_size,
                dims.q_num_heads,
                dims.q_sequence_length,
                dims.v_head_size,
            ],
            r => panic!("Attention: unexpected rank {r}"),
        }),
    });

    if let Some(present_key) = node.outputs.get_mut(1) {
        present_key.ty = ArgType::Tensor(TensorType {
            elem_type: node.inputs[4].ty.elem_type().clone(),
            rank: 4,
            static_shape: Some(vec![
                dims.batch_size,
                dims.kv_num_heads,
                dims.total_sequence_length,
                dims.head_size,
            ]),
        });
    }

    if let Some(present_value) = node.outputs.get_mut(2) {
        present_value.ty = ArgType::Tensor(TensorType {
            elem_type: node.inputs[5].ty.elem_type().clone(),
            rank: 4,
            static_shape: Some(vec![
                dims.batch_size,
                dims.kv_num_heads,
                dims.total_sequence_length,
                dims.v_head_size,
            ]),
        });
    }

    if let Some(qk_matmul_output) = node.outputs.get_mut(3) {
        qk_matmul_output.ty = ArgType::Tensor(TensorType {
            elem_type: node.inputs[0].ty.elem_type().clone(),
            rank: 4,
            static_shape: Some(vec![
                dims.batch_size,
                dims.q_num_heads,
                dims.q_sequence_length,
                dims.total_sequence_length,
            ]),
        });
    }
}

#[derive(Debug, Clone)]
pub struct AttentionDims {
    pub rank: usize,
    pub batch_size: usize,
    pub head_size: usize,
    pub q_num_heads: usize,
    pub kv_num_heads: usize,
    pub q_sequence_length: usize,
    pub kv_sequence_length: usize,
    pub q_hidden_size: usize,
    pub k_hidden_size: usize,
    pub v_hidden_size: usize,
    pub v_head_size: usize,
    pub total_sequence_length: usize,
}

impl AttentionDims {
    pub fn attn_mask_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.q_num_heads,
            self.q_sequence_length,
            self.total_sequence_length,
        ]
    }
}

fn compute_dims(node: &Node) -> AttentionDims {
    if node.inputs.len() < 3 {
        panic!("Attention must have at least 3 inputs")
    }
    if node.outputs.is_empty() {
        panic!("Attention must have at least 1 output")
    }

    let q = extract_tensor(node.inputs.first(), "Q").unwrap();
    let k = extract_tensor(node.inputs.get(1), "K").unwrap();
    let v = extract_tensor(node.inputs.get(2), "V").unwrap();
    require_float_type(q, "Q");
    require_float_type(k, "K");
    require_float_type(v, "V");
    let (q_shape, q_rank) = extract_shape_and_rank(q, "Q");
    let (k_shape, k_rank) = extract_shape_and_rank(k, "K");
    let (v_shape, v_rank) = extract_shape_and_rank(v, "V");

    if !all_eq(&[q_rank, k_rank, v_rank]) {
        panic!(
            "Attention: Q, K, V parameters must have the same dimension, but are {q_rank}D, {k_rank}D, {v_rank}D respectively."
        )
    }

    let (q_batch, q_heads, q_seq, q_head) = maybe_4d_shape(&q_shape, "Q");
    let (k_batch, k_heads, k_seq, k_head) = maybe_4d_shape(&k_shape, "K");
    let (v_batch, v_heads, v_seq, v_head) = maybe_4d_shape(&v_shape, "V");
    if !all_eq(&[q_batch, k_batch, v_batch]) {
        panic!(
            "Attention: Q, K, V parameters must agree on batch_size. Got {q_batch}, {k_batch}, {v_batch} respectively."
        )
    }
    if q_rank == 4 && q_head != k_head {
        panic!(
            "Attention: Q, K parameters must agree on head_size. Got {q_head}, {k_head} respectively."
        )
    }
    if let Some((k_heads, v_heads)) = k_heads.zip(v_heads)
        && k_heads != v_heads
    {
        panic!(
            "Attention: K, V parameters must agree on kv_num_heads. Got {k_heads}, {v_heads} respectively."
        )
    }
    if k_seq != v_seq {
        panic!(
            "Attention: K, V parameters must agree on kv_sequence_length. Got {k_seq}, {v_seq} respectively."
        )
    }

    let q_sequence_length = q_seq;
    let kv_sequence_length = k_seq;
    let v_head_size = v_head;

    let mut kv_num_heads = None;
    let mut q_num_heads = None;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "kv_num_heads" => kv_num_heads = Some(value.clone().into_i64() as usize),
            "q_num_heads" => q_num_heads = Some(value.clone().into_i64() as usize),
            _ => (),
        }
    }

    let q_num_heads = match q_heads {
        Some(v) => v,
        None => {
            q_num_heads.expect("Attention: the q_num_heads attribute must be present for 3D inputs")
        }
    };
    let kv_num_heads = match k_heads {
        Some(v) => v,
        None => kv_num_heads
            .expect("Attention: the kv_num_heads attribute must be present for 3D inputs"),
    };
    let head_size = if q_rank == 3 {
        let q_head_size = q_head / q_num_heads;
        let k_head_size = k_head / kv_num_heads;
        if q_head_size != k_head_size {
            panic!(
                "Attention: Q, K parameters (3D) must agree on computed head_size. Computed {q_head_size}, {k_head_size} respectively"
            )
        }

        q_head_size
    } else {
        q_head
    };

    let mut total_sequence_length = kv_sequence_length;

    if node.inputs.len() > 3 || node.outputs.len() > 1 {
        let attn_mask = extract_tensor(node.inputs.get(3), "attn_mask")
            .expect("Attention: attn_mask must be present if cache is used");
        let (mask_shape, mask_rank) = extract_shape_and_rank(attn_mask, "attn_mask");
        let [mask_seq, mask_total_seq] = mask_shape.as_slice() else {
            panic!("Attention: attn_mask must be a 2D tensor, but is {mask_rank}D instead")
        };
        if *mask_seq != q_sequence_length {
            panic!(
                "Attention attn_mask shape does not agree with q_sequence_len: {mask_seq} != {q_sequence_length}"
            )
        }

        if node.inputs.len() > 4 || node.outputs.len() > 1 {
            let pk1 = extract_tensor(node.inputs.get(4), "past_key")
                .expect("Attention: past_key must be present if cache is used");
            let (pk1_shape, pk1_rank) = extract_shape_and_rank(pk1, "past_key");
            let [pk1_batch, pk1_heads, pk1_seq, pk1_head] = pk1_shape.as_slice() else {
                panic!("Attention: past_key must be a 4D tensor, but is {pk1_rank}D instead")
            };
            let pv1 = extract_tensor(node.inputs.get(5), "past_value")
                .expect("Attention: past_value must be present if cache is used");
            let (pv1_shape, pv1_rank) = extract_shape_and_rank(pv1, "past_value");
            let [pv1_batch, pv1_heads, pv1_seq, pv1_head] = pv1_shape.as_slice() else {
                panic!("Attention: past_value must be a 4D tensor, but is {pv1_rank}D instead")
            };
            extract_tensor(node.outputs.get(1), "present_key")
                .expect("Attention: present_key must be present if cache is used");
            extract_tensor(node.outputs.get(2), "present_value")
                .expect("Attention: present_value must be present if cache is used");

            if !all_eq(&[*pk1_batch, *pv1_batch, q_batch]) {
                panic!(
                    "Attention: past_[key,value] have invalid batch_size. Got {pk1_batch}, {pv1_batch}, but expected {q_batch}"
                )
            }
            if !all_eq(&[*pk1_heads, *pv1_heads, kv_num_heads]) {
                panic!(
                    "Attention: past_[key,value] have invalid kv_num_heads. Got {pk1_heads}, {pv1_heads}, but expected {kv_num_heads}"
                )
            }
            if pk1_seq != pv1_seq {
                panic!(
                    "Attention: past_[key,value] need to agree on past_sequence_length. Got {pk1_seq}, {pv1_seq} respectively"
                )
            }
            total_sequence_length = pk1_seq + kv_sequence_length;
            if *pk1_head != head_size {
                panic!(
                    "Attention: past_key has invalid head_size. Got {pk1_head}, but expected {head_size}"
                );
            }
            if *pv1_head != v_head_size {
                panic!(
                    "Attention: past_value has invalid v_head_size. Got {pv1_head}, but expected {v_head_size}"
                );
            }
        }

        if *mask_total_seq != total_sequence_length {
            panic!(
                "Attention attn_mask shape does not agree with total_sequence_length: {mask_seq} != {total_sequence_length}"
            )
        }
    }

    AttentionDims {
        rank: q_rank,
        batch_size: q_batch,
        head_size,
        q_num_heads,
        kv_num_heads,
        q_sequence_length,
        kv_sequence_length,
        q_hidden_size: q_num_heads * head_size,
        k_hidden_size: kv_num_heads * head_size,
        v_hidden_size: kv_num_heads * v_head_size,
        v_head_size,
        total_sequence_length,
    }
}

fn extract_tensor<'a>(arg: Option<&'a Argument>, name: &str) -> Option<&'a TensorType> {
    match &arg?.ty {
        ArgType::Tensor(v) => Some(v),
        _ => panic!("Attention: {name} input must be a tensor"),
    }
}

fn extract_shape_and_rank(tensor: &TensorType, name: &str) -> (Vec<usize>, usize) {
    let Some(shape) = tensor.static_shape.as_ref() else {
        panic!("Attention: {name} tensor must have a static shape")
    };
    if tensor.rank != shape.len() {
        panic!("Attention: the rank and size of the static shape of {name} do not match")
    }
    (shape.clone(), tensor.rank)
}

fn require_float_type(ty: &TensorType, name: &str) {
    if !matches!(
        ty.elem_type,
        ElementType::Float16 | ElementType::Float32 | ElementType::Float64
    ) {
        panic!("Attention: {name} must be a float tensor")
    }
}

fn maybe_4d_shape(shape: &[usize], param: &str) -> (usize, Option<usize>, usize, usize) {
    match shape {
        [batch_size, num_heads, seq_len, head_size] => {
            (*batch_size, Some(*num_heads), *seq_len, *head_size)
        }
        [batch_size, seq_len, head_size] => (*batch_size, None, *seq_len, *head_size),
        _ => panic!(
            "Attention: {param} parameter must be a 3D or 4D tensor, but got {}D tensor instead",
            shape.len()
        ),
    }
}

fn all_eq<T: Eq>(vals: &[T]) -> bool {
    let Some(first) = vals.first() else {
        return true;
    };
    vals.iter().all(|v| v == first)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
mod tests {
    use super::*;
    use crate::{NodeType, node::test_utils::NodeBuilder};
    use rstest::rstest;

    fn create_test_node(
        q: Option<(usize, Option<&[usize]>)>,
        k: Option<(usize, Option<&[usize]>)>,
        v: Option<(usize, Option<&[usize]>)>,
        attn_mask: Option<(ElementType, usize, Option<&[usize]>)>,
        past_key: Option<(usize, Option<&[usize]>)>,
        past_value: Option<(usize, Option<&[usize]>)>,
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

        if let Some((rank, shape)) = q {
            builder = builder.input_tensor_f32("q", rank, shape.map(|v| v.to_vec()));
        }
        if let Some((rank, shape)) = k {
            builder = builder.input_tensor_f32("k", rank, shape.map(|v| v.to_vec()));
        }
        if let Some((rank, shape)) = v {
            builder = builder.input_tensor_f32("v", rank, shape.map(|v| v.to_vec()));
        }
        if let Some((ty, rank, shape)) = attn_mask {
            builder = builder.add_input(
                "attn_mask",
                ArgType::Tensor(TensorType {
                    elem_type: ty,
                    rank,
                    static_shape: shape.map(|v| v.to_vec()),
                }),
            );
        }
        if let Some((rank, shape)) = past_key {
            builder = builder.input_tensor_f32("past_key", rank, shape.map(|v| v.to_vec()));
        }
        if let Some((rank, shape)) = past_value {
            builder = builder.input_tensor_f32("past_value", rank, shape.map(|v| v.to_vec()));
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

    fn create_test_node_no_attributes(
        q: Option<(usize, Option<&[usize]>)>,
        k: Option<(usize, Option<&[usize]>)>,
        v: Option<(usize, Option<&[usize]>)>,
        attn_mask: Option<(ElementType, usize, Option<&[usize]>)>,
        past_key: Option<(usize, Option<&[usize]>)>,
        past_value: Option<(usize, Option<&[usize]>)>,
        y: Option<usize>,
        present_key: Option<usize>,
        present_value: Option<usize>,
        qk_matmul_output: Option<usize>,
    ) -> Node {
        create_test_node(
            q,
            k,
            v,
            attn_mask,
            past_key,
            past_value,
            y,
            present_key,
            present_value,
            qk_matmul_output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[rstest]
    // Missing static shape
    #[case(Some((4,None)), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,None)), Some((4,Some(&[1,1,2,2][..]))), None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,None)), None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((ElementType::Bool,4,None)), None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, Some((4,None)), None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, Some((4,None)), Some(4), None, None, None)]
    // Missing required inputs or outputs
    #[case(None, Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), None, Some((4,Some(&[1,1,2,2][..]))), None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, None, None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, None, None, None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((ElementType::Bool,2,Some(&[2,2][..]))), Some((2,Some(&[2,2][..]))), None, Some(4), None, None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), None, None, None, Some(4), Some(2), None, None)]
    #[case(Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((4,Some(&[1,1,2,2][..]))), Some((ElementType::Bool,2,Some(&[2,2][..]))), Some((2,Some(&[2,2][..]))), Some((2,Some(&[2,2][..]))), Some(4), None, None, None)]
    #[should_panic]
    fn test_fail_on_invalid_inputs(
        #[case] q: Option<(usize, Option<&[usize]>)>,
        #[case] k: Option<(usize, Option<&[usize]>)>,
        #[case] v: Option<(usize, Option<&[usize]>)>,
        #[case] attn_mask: Option<(ElementType, usize, Option<&[usize]>)>,
        #[case] past_key: Option<(usize, Option<&[usize]>)>,
        #[case] past_value: Option<(usize, Option<&[usize]>)>,
        #[case] y: Option<usize>,
        #[case] present_key: Option<usize>,
        #[case] present_value: Option<usize>,
        #[case] qk_matmul_output: Option<usize>,
    ) {
        let node = create_test_node_no_attributes(
            q,
            k,
            v,
            attn_mask,
            past_key,
            past_value,
            y,
            present_key,
            present_value,
            qk_matmul_output,
        );
        attention_config(&node);
    }

    #[test]
    fn test_simple_4d() {
        let node = create_test_node_no_attributes(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        assert!(!config.is_causal);
        assert_eq!(
            config.qk_matmul_output_mode,
            AttentionQkMatmulOutputMode::Matmul
        );
        assert!((config.scale - 0.7071067811865475).abs() < f64::EPSILON);
        assert_eq!(config.softcap, 0.0);
        assert_eq!(config.softmax_precision, None);
        let dims = &config.computed_dims;
        assert_eq!(dims.rank, 4);
        assert_eq!(dims.batch_size, 1);
        assert_eq!(dims.head_size, 2);
        assert_eq!(dims.q_num_heads, 1);
        assert_eq!(dims.kv_num_heads, 1);
        assert_eq!(dims.q_sequence_length, 2);
        assert_eq!(dims.kv_sequence_length, 2);
        assert_eq!(dims.q_hidden_size, 2);
        assert_eq!(dims.k_hidden_size, 2);
        assert_eq!(dims.v_hidden_size, 2);
        assert_eq!(dims.v_head_size, 2);
        assert_eq!(dims.total_sequence_length, 2);
    }

    #[test]
    fn test_simple_3d() {
        let node = create_test_node(
            Some((3, Some(&[1, 2, 2][..]))),
            Some((3, Some(&[1, 2, 2][..]))),
            Some((3, Some(&[1, 2, 2][..]))),
            None,
            None,
            None,
            Some(3),
            None,
            None,
            None,
            None,
            Some(1),
            Some(1),
            None,
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        assert!((config.scale - 0.7071067811865475).abs() < f64::EPSILON);
        let dims = &config.computed_dims;
        assert_eq!(dims.rank, 3);
        assert_eq!(dims.batch_size, 1);
        assert_eq!(dims.head_size, 2);
        assert_eq!(dims.q_num_heads, 1);
        assert_eq!(dims.kv_num_heads, 1);
        assert_eq!(dims.q_sequence_length, 2);
        assert_eq!(dims.kv_sequence_length, 2);
        assert_eq!(dims.q_hidden_size, 2);
        assert_eq!(dims.k_hidden_size, 2);
        assert_eq!(dims.v_hidden_size, 2);
        assert_eq!(dims.v_head_size, 2);
        assert_eq!(dims.total_sequence_length, 2);
    }

    #[test]
    fn test_attn_mask() {
        let node = create_test_node_no_attributes(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((ElementType::Bool, 2, Some(&[2, 2][..]))),
            None,
            None,
            Some(4),
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        let dims = &config.computed_dims;
        assert_eq!(dims.rank, 4);
        assert_eq!(dims.batch_size, 1);
        assert_eq!(dims.head_size, 2);
        assert_eq!(dims.q_num_heads, 1);
        assert_eq!(dims.kv_num_heads, 1);
        assert_eq!(dims.q_sequence_length, 2);
        assert_eq!(dims.kv_sequence_length, 2);
        assert_eq!(dims.q_hidden_size, 2);
        assert_eq!(dims.k_hidden_size, 2);
        assert_eq!(dims.v_hidden_size, 2);
        assert_eq!(dims.v_head_size, 2);
        assert_eq!(dims.total_sequence_length, 2);
    }

    #[test]
    fn test_softcap() {
        let node = create_test_node(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(2.0),
            None,
        );
        let config = attention_config(&node);
        assert_eq!(config.softcap, 2.0);
    }

    #[test]
    fn test_cache() {
        let node = create_test_node(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 1, 2][..]))),
            Some((4, Some(&[1, 1, 1, 2][..]))),
            Some((ElementType::Bool, 2, Some(&[2, 2][..]))),
            Some((4, Some(&[1, 1, 1, 2][..]))),
            Some((4, Some(&[1, 1, 1, 2][..]))),
            Some(4),
            Some(4),
            Some(4),
            None,
            None,
            Some(1),
            Some(1),
            None,
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        let dims = &config.computed_dims;
        assert_eq!(dims.rank, 4);
        assert_eq!(dims.batch_size, 1);
        assert_eq!(dims.head_size, 2);
        assert_eq!(dims.q_num_heads, 1);
        assert_eq!(dims.kv_num_heads, 1);
        assert_eq!(dims.q_sequence_length, 2);
        assert_eq!(dims.kv_sequence_length, 1);
        assert_eq!(dims.q_hidden_size, 2);
        assert_eq!(dims.k_hidden_size, 2);
        assert_eq!(dims.v_hidden_size, 2);
        assert_eq!(dims.v_head_size, 2);
        assert_eq!(dims.total_sequence_length, 2);
    }

    #[test]
    fn test_custom_scale() {
        let node = create_test_node(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(2.0),
            None,
            None,
        );
        let config = attention_config(&node);
        assert_eq!(config.scale, 2.0);
    }

    #[test]
    fn test_is_causal() {
        let node = create_test_node(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        assert!(config.is_causal);
    }

    #[rstest]
    #[case(0, AttentionQkMatmulOutputMode::Matmul)]
    #[case(1, AttentionQkMatmulOutputMode::MatmulPlusAttentionMask)]
    #[case(2, AttentionQkMatmulOutputMode::MatmulAfterSoftcap)]
    #[case(3, AttentionQkMatmulOutputMode::MatmulAfterSoftmax)]
    fn test_qk_matmul_output(#[case] raw: i64, #[case] mode: AttentionQkMatmulOutputMode) {
        let node = create_test_node(
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            Some((4, Some(&[1, 1, 2, 2][..]))),
            None,
            None,
            None,
            Some(4),
            None,
            None,
            None,
            Some(1),
            None,
            None,
            Some(raw),
            None,
            None,
            None,
        );
        let config = attention_config(&node);
        assert_eq!(config.qk_matmul_output_mode, mode);
    }
}
