use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::attention::{AttentionConfig, AttentionQkMatmulOutputMode};
use quote::quote;

#[derive(Debug, Clone)]
pub struct AttentionNode {
    pub inputs: AttentionNodeInputs,
    pub outputs: AttentionNodeOutputs,
    pub config: AttentionConfig,
}

impl AttentionNode {
    pub fn new(
        inputs: AttentionNodeInputs,
        outputs: AttentionNodeOutputs,
        config: AttentionConfig,
    ) -> Self {
        Self {
            inputs,
            outputs,
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionNodeInputs {
    pub q: TensorType,
    pub k: TensorType,
    pub v: TensorType,
    pub attn_mask: Option<TensorType>,
    pub past_key: Option<TensorType>,
    pub past_value: Option<TensorType>,
}

impl AttentionNodeInputs {
    pub fn new(
        q: TensorType,
        k: TensorType,
        v: TensorType,
        attn_mask: Option<TensorType>,
        past_key: Option<TensorType>,
        past_value: Option<TensorType>,
    ) -> Self {
        Self {
            q,
            k,
            v,
            attn_mask,
            past_key,
            past_value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionNodeOutputs {
    pub y: TensorType,
    pub present_key: Option<TensorType>,
    pub present_value: Option<TensorType>,
    pub qk_matmul_output: Option<TensorType>,
}

impl AttentionNodeOutputs {
    pub fn new(
        y: TensorType,
        present_key: Option<TensorType>,
        present_value: Option<TensorType>,
        qk_matmul_output: Option<TensorType>,
    ) -> Self {
        Self {
            y,
            present_key,
            present_value,
            qk_matmul_output,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AttentionNode {
    fn input_types(&self) -> Vec<Type> {
        let mut v = vec![
            Type::Tensor(self.inputs.q.clone()),
            Type::Tensor(self.inputs.k.clone()),
            Type::Tensor(self.inputs.v.clone()),
        ];
        if let Some(input_attn_mask) = self.inputs.attn_mask.clone() {
            v.push(Type::Tensor(input_attn_mask));
        }
        if let Some(input_past_key) = self.inputs.past_key.clone() {
            v.push(Type::Tensor(input_past_key));
        }
        if let Some(input_past_value) = self.inputs.past_value.clone() {
            v.push(Type::Tensor(input_past_value));
        }
        v
    }

    fn output_types(&self) -> Vec<Type> {
        let mut v = vec![Type::Tensor(self.outputs.y.clone())];
        if let Some(output_present_key) = self.outputs.present_key.clone() {
            v.push(Type::Tensor(output_present_key));
        }
        if let Some(output_present_value) = self.outputs.present_value.clone() {
            v.push(Type::Tensor(output_present_value));
        }
        if let Some(output_qk_matmul_output) = self.outputs.qk_matmul_output.clone() {
            v.push(Type::Tensor(output_qk_matmul_output));
        }
        v
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> proc_macro2::TokenStream {
        // For the description of the algorithm, see ONNX docs (https://onnx.ai/onnx/operators/onnx__Attention.html)
        // or the reference implementation in onnx/reference/ops/op_attention.py

        let past_kv = match (
            self.inputs.past_key.as_ref(),
            self.inputs.past_value.as_ref(),
        ) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => panic!("Attention: past_key and past_value must be used together."),
        };
        let present_kv = match (
            self.outputs.present_key.as_ref(),
            self.outputs.present_value.as_ref(),
        ) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => panic!("Attention: present_key and present_value must be used together."),
        };

        let rank = self.inputs.q.rank;

        let q = scope.tensor_use_owned(&self.inputs.q, node_position);
        let k = scope.tensor_use_owned(&self.inputs.k, node_position);
        let v = scope.tensor_use_owned(&self.inputs.v, node_position);
        let output_y = &self.outputs.y.name;

        let mut body = proc_macro2::TokenStream::new();
        body.extend(quote! {
            let q = #q;
            let k = #k;
            let v = #v;
        });

        let scale = self.config.scale.map(|scale| {
            let scale = scale.sqrt();
            quote! {
                let scale = #scale;
            }
        });

        // Reshape the qk input if they are only 3D tensors
        let mut reshape_output = quote! {};
        if rank == 3 {
            let kv_num_heads = self.config.kv_num_heads;
            let q_num_heads = self.config.q_num_heads;

            body.extend(quote! {
                let [batch_size, q_sequence_length, q_hidden_size] = q.dims();
                #[allow(clippy::identity_op)] // q_num_heads could be 1
                let head_size = q_hidden_size / #q_num_heads;
                let kv_sequence_length = k.dims()[1];
                #[allow(clippy::identity_op)] // kv_num_heads could be 1
                let v_head_size = v.dims()[2] / #kv_num_heads;
                let q = q.reshape([batch_size, q_sequence_length, #q_num_heads, head_size])
                        .permute([0, 2, 1, 3]);
                let k = k.reshape([batch_size, kv_sequence_length, #kv_num_heads, head_size])
                        .permute([0, 2, 1, 3]);
                let v = v.reshape([batch_size, kv_sequence_length, #kv_num_heads, v_head_size])
                        .permute([0, 2, 1, 3]);
            });
            body.extend(scale.unwrap_or_else(|| {
                quote! {
                    let scale = (1.0 / (head_size as f64).sqrt()).sqrt();
                }
            }));

            reshape_output = quote! {
                let #output_y = #output_y.permute([0, 2, 1, 3]).reshape([batch_size as i32, q_sequence_length as i32, -1]);
            };
        } else {
            body.extend(scale.unwrap_or_else(|| {
                quote! {
                    let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                }
            }));
        }

        match (past_kv, present_kv) {
            (Some((past_k, past_v)), Some((present_k, present_v))) => {
                let past_k = scope.tensor_use_owned(past_k, node_position);
                let past_v = scope.tensor_use_owned(past_v, node_position);
                let present_k = &present_k.name;
                let present_v = &present_v.name;

                body.extend(quote! {
                    let #present_k = Tensor::cat([#past_k, k].to_vec(), 2);
                    let k = #present_k.clone();
                    let #present_v = Tensor::cat([#past_v, v].to_vec(), 2);
                    let v = #present_v.clone();
                });
            }
            (None, None) => (),
            _ => {
                panic!("Attention: past_[key,value] and present_[key,value] must be used together.")
            }
        }

        if self.inputs.attn_mask.is_some() || self.config.is_causal {
            body.extend(quote! {
                let q_dims = q.dims();
                let k_dims = k.dims();
            });
        }

        let qk = quote! { qk };
        let attn_mask_shape = quote! {{
            let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
            [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
        }};
        let mut attn_mask = match self.inputs.attn_mask.as_ref() {
            Some(mask) => {
                let mask_input = scope.tensor_use_owned(mask, node_position);
                let mask = match mask.kind {
                    TensorKind::Int => quote! { #mask_input.float() },
                    TensorKind::Float => mask_input,
                    TensorKind::Bool => {
                        quote! {{
                            let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &#mask_input.device());
                            float_mask.mask_fill(#mask_input.bool_not(), f32::NEG_INFINITY)
                        }}
                    }
                };

                quote! {
                    let shape = #attn_mask_shape;
                    let #qk = #qk + #mask.expand::<4, _>(shape);
                }
            }
            None => {
                quote! {}
            }
        };
        if self.config.is_causal {
            attn_mask = quote! {
                let #qk = {
                    let shape = #attn_mask_shape;
                    let mask = Tensor::<B, 2>::ones([shape[2], shape[3]], &#qk.device());
                    let mask = mask.tril(0).bool().bool_not();
                    let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &mask.device()).mask_fill(mask, f32::NEG_INFINITY);
                    #qk + float_mask.expand::<4, _>(shape)
                };
            };
        }

        let capped = quote! { capped };
        let (mut qk_matmul_a, mut qk_matmul_b, mut qk_matmul_c, mut qk_matmul_d) =
            (quote! {}, quote! {}, quote! {}, quote! {});
        if let Some(out) = self.outputs.qk_matmul_output.as_ref().map(|v| &v.name) {
            match self.config.qk_matmul_output_mode {
                AttentionQkMatmulOutputMode::Matmul => {
                    qk_matmul_a = quote! {
                        let #out = #qk.clone();
                    };
                }
                AttentionQkMatmulOutputMode::MatmulPlusAttentionMask => {
                    qk_matmul_b = quote! {
                        let #out = #qk.clone();
                    };
                }
                AttentionQkMatmulOutputMode::MatmulAfterSoftcap => {
                    qk_matmul_c = quote! {
                        let #out = #capped.clone();
                    };
                }
                AttentionQkMatmulOutputMode::MatmulAfterSoftmax => {
                    qk_matmul_d = quote! {
                        let #out = scores.clone();
                    };
                }
            }
        }

        let softcap = match self.config.softcap {
            softcap if softcap != 0.0 => {
                let inv_softcap = 1.0 / softcap;
                // Implemented according to https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations
                quote! {
                    let #capped = {
                        let score = #qk * #inv_softcap;
                        let score = score.tanh();
                        score * #softcap
                    };
                    #qk_matmul_c
                }
            }
            _ => quote! {
                let #capped = #qk;
            },
        };

        if self.config.softmax_precision.is_some() {
            panic!("Attention: non-default softmax precision is not yet supported")
        }

        let mut output = vec![output_y];
        match present_kv {
            Some((a, b)) => output.extend_from_slice(&[&a.name, &b.name]),
            None => (),
        }
        if let Some(t) = self.outputs.qk_matmul_output.as_ref() {
            output.push(&t.name);
        }
        let output = quote! { (#(#output,)*) };

        quote! {
            let #output = {
                #body

                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let #qk = q_scaled.matmul(k_transpose);
                #qk_matmul_a
                #attn_mask
                #qk_matmul_b
                #softcap
                let scores = softmax(#capped, 3);
                #qk_matmul_d
                let #output_y = scores.matmul(v);
                #reshape_output
                #output
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::activation::softmax");
    }

    fn into_node(self) -> Node<PS> {
        Node::Attention(self)
    }
}

impl OnnxIntoNode for AttentionNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let q = TensorType::from(node.inputs.first().unwrap());
        let k = TensorType::from(node.inputs.get(1).unwrap());
        let v = TensorType::from(node.inputs.get(2).unwrap());
        let attn_mask = node.inputs.get(3).map(TensorType::from);
        let past_key = node.inputs.get(4).map(TensorType::from);
        let past_value = node.inputs.get(5).map(TensorType::from);
        let y = TensorType::from(node.outputs.first().unwrap());
        let present_key = node.outputs.get(1).map(TensorType::from);
        let present_value = node.outputs.get(2).map(TensorType::from);
        let qk_matmul_output = node.outputs.get(3).map(TensorType::from);
        let config = onnx_ir::node::attention::attention_config(&node);

        AttentionNode::new(
            AttentionNodeInputs::new(q, k, v, attn_mask, past_key, past_value),
            AttentionNodeOutputs::new(y, present_key, present_value, qk_matmul_output),
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::node::tests::one_node_graph;

    #[test]
    fn test_attention_codegen_simple_4d() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    None,
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_simple_3d() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 3),
                    TensorType::new_float("k", 3),
                    TensorType::new_float("v", 3),
                    None,
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 3), None, None, None),
                AttentionConfig::new(
                    false,
                    Some(1),
                    Some(1),
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let [batch_size, q_sequence_length, q_hidden_size] = q.dims();
                        #[allow(clippy::identity_op)]
                        let head_size = q_hidden_size / 1usize;
                        let kv_sequence_length = k.dims()[1];
                        #[allow(clippy::identity_op)]
                        let v_head_size = v.dims()[2] / 1usize;
                        let q = q
                            .reshape([batch_size, q_sequence_length, 1usize, head_size])
                            .permute([0, 2, 1, 3]);
                        let k = k
                            .reshape([batch_size, kv_sequence_length, 1usize, head_size])
                            .permute([0, 2, 1, 3]);
                        let v = v
                            .reshape([batch_size, kv_sequence_length, 1usize, v_head_size])
                            .permute([0, 2, 1, 3]);
                        let scale = (1.0 / (head_size as f64).sqrt()).sqrt();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        let y = y.permute([0, 2, 1, 3]).reshape([batch_size as i32, q_sequence_length as i32, -1]);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_attn_mask_bool() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    Some(TensorType::new_bool("attn_mask", 2)),
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, attn_mask: Tensor<B, 2, Bool>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_dims = q.dims();
                        let k_dims = k.dims();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let shape = {
                            let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                            [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                        };
                        let qk = qk + {
                            let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &attn_mask.device());
                            float_mask.mask_fill(attn_mask.bool_not(), f32::NEG_INFINITY)
                        }
                        .expand::<4, _>(shape);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into(), "attn_mask".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_attn_mask_int() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    Some(TensorType::new_int("attn_mask", 2)),
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, attn_mask: Tensor<B, 2, Int>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_dims = q.dims();
                        let k_dims = k.dims();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let shape = {
                            let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                            [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                        };
                        let qk = qk + attn_mask.float().expand::<4, _>(shape);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into(), "attn_mask".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_attn_mask_float() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    Some(TensorType::new_float("attn_mask", 2)),
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, attn_mask: Tensor<B, 2>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_dims = q.dims();
                        let k_dims = k.dims();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let shape = {
                            let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                            [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                        };
                        let qk = qk + attn_mask.expand::<4, _>(shape);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into(), "attn_mask".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_softcap() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    None,
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    2.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let capped = {
                            let score = qk * 0.5f64;
                            let score = score.tanh();
                            score * 2f64
                        };
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into()],
            vec!["y".into()],
        );
    }

    #[test]
    fn test_attention_codegen_cache() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    Some(TensorType::new_bool("attn_mask", 2)),
                    Some(TensorType::new_float("past_key", 4)),
                    Some(TensorType::new_float("past_value", 4)),
                ),
                AttentionNodeOutputs::new(
                    TensorType::new_float("y", 4),
                    Some(TensorType::new_float("present_key", 4)),
                    Some(TensorType::new_float("present_value", 4)),
                    None,
                ),
                AttentionConfig::new(
                    false,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, attn_mask: Tensor<B, 2, Bool>,
                    past_key: Tensor<B, 4>, past_value: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
                    let (y, present_key, present_value) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let present_key = Tensor::cat([past_key, k].to_vec(), 2);
                        let k = present_key.clone();
                        let present_value = Tensor::cat([past_value, v].to_vec(), 2);
                        let v = present_value.clone();
                        let q_dims = q.dims();
                        let k_dims = k.dims();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let shape = {
                            let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                            [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                        };
                        let qk = qk + {
                            let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &attn_mask.device());
                            float_mask.mask_fill(attn_mask.bool_not(), f32::NEG_INFINITY)
                        }
                        .expand::<4, _>(shape);
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y, present_key, present_value)
                    };
                    (y, present_key, present_value)
                }
            },
            vec![
                "q".into(),
                "k".into(),
                "v".into(),
                "attn_mask".into(),
                "past_key".into(),
                "past_value".into(),
            ],
            vec!["y".into(), "present_key".into(), "present_value".into()],
        );
    }

    #[test]
    fn test_attention_codegen_is_causal() {
        one_node_graph(
            AttentionNode::new(
                AttentionNodeInputs::new(
                    TensorType::new_float("q", 4),
                    TensorType::new_float("k", 4),
                    TensorType::new_float("v", 4),
                    None,
                    None,
                    None,
                ),
                AttentionNodeOutputs::new(TensorType::new_float("y", 4), None, None, None),
                AttentionConfig::new(
                    true,
                    None,
                    None,
                    AttentionQkMatmulOutputMode::Matmul,
                    None,
                    0.0,
                    None,
                ),
            ),
            quote! {
                pub fn forward(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
                    let (y,) = {
                        let q = q;
                        let k = k;
                        let v = v;
                        let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                        let q_dims = q.dims();
                        let k_dims = k.dims();
                        let q_scaled = q * scale;
                        let k_scaled = k * scale;
                        let k_transpose = k_scaled.transpose();
                        let qk = q_scaled.matmul(k_transpose);
                        let qk = {
                            let shape = {
                                let [batch_size, q_num_heads, q_sequence_length, _] = q_dims;
                                [batch_size, q_num_heads, q_sequence_length, k_dims[2]]
                            };
                            let mask = Tensor::<B, 2>::ones([shape[2], shape[3]], &qk.device());
                            let mask = mask.tril(0).bool().bool_not();
                            let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &mask.device())
                                .mask_fill(mask, f32::NEG_INFINITY);
                            qk + float_mask.expand::<4, _>(shape)
                        };
                        let capped = qk;
                        let scores = softmax(capped, 3);
                        let y = scores.matmul(v);
                        (y,)
                    };
                    y
                }
            },
            vec!["q".into(), "k".into(), "v".into()],
            vec!["y".into()],
        );
    }
}
