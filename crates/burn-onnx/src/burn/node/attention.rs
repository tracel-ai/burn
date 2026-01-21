use super::prelude::*;
use onnx_ir::node::attention::AttentionQkMatmulOutputMode;

impl NodeCodegen for onnx_ir::attention::AttentionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // For the description of the algorithm, see ONNX docs (https://onnx.ai/onnx/operators/onnx__Attention.html)
        // or the reference implementation in onnx/reference/ops/op_attention.py

        // Get Q, K, V inputs (required) - these may have any names, we normalize to q, k, v
        let q = scope.arg(self.inputs.first().unwrap());
        let k = scope.arg(self.inputs.get(1).unwrap());
        let v = scope.arg(self.inputs.get(2).unwrap());

        // Get output names
        let output_y = arg_to_ident(self.outputs.first().unwrap());

        // Check for past/present key-value pairs
        let past_kv = match (self.inputs.get(4), self.inputs.get(5)) {
            (Some(_), Some(_)) => true,
            (None, None) => false,
            _ => panic!("Attention: past_key and past_value must be used together."),
        };
        let present_kv = match (self.outputs.get(1), self.outputs.get(2)) {
            (Some(_), Some(_)) => true,
            (None, None) => false,
            _ => panic!("Attention: present_key and present_value must be used together."),
        };

        // Get rank from Q input
        let rank = match &self.inputs.first().unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for Q"),
        };

        let mut body = TokenStream::new();

        // Normalize input names to q, k, v for consistent use in generated code
        body.extend(quote! {
            let q = #q;
            let k = #k;
            let v = #v;
        });

        // Handle scale
        let scale = self.config.scale.map(|scale| {
            let scale = scale.sqrt();
            quote! {
                let scale = #scale;
            }
        });

        // Reshape the qkv inputs if they are only 3D tensors
        let mut reshape_output = quote! {};
        if rank == 3 {
            let kv_num_heads = self
                .config
                .kv_num_heads
                .expect("kv_num_heads required for rank 3");
            let q_num_heads = self
                .config
                .q_num_heads
                .expect("q_num_heads required for rank 3");

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

        // Handle past/present key-value caching
        if past_kv && present_kv {
            let past_k = scope.arg(self.inputs.get(4).unwrap());
            let past_v = scope.arg(self.inputs.get(5).unwrap());
            let present_k = arg_to_ident(self.outputs.get(1).unwrap());
            let present_v = arg_to_ident(self.outputs.get(2).unwrap());

            body.extend(quote! {
                let #present_k = Tensor::cat([#past_k, k].to_vec(), 2);
                let k = #present_k.clone();
                let #present_v = Tensor::cat([#past_v, v].to_vec(), 2);
                let v = #present_v.clone();
            });
        } else if past_kv != present_kv {
            panic!("Attention: past_[key,value] and present_[key,value] must be used together.")
        }

        // Handle attention mask or causal masking
        if self.inputs.get(3).is_some() || self.config.is_causal {
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

        // Handle attention mask input
        let mut attn_mask = if let Some(mask_input) = self.inputs.get(3) {
            let mask_arg = scope.arg(mask_input);
            let mask = match &mask_input.ty {
                onnx_ir::ir::ArgType::Tensor(t) => match &t.dtype {
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        quote! { #mask_arg.float() }
                    }
                    dtype if dtype.is_float() => mask_arg,
                    dtype if dtype.is_bool() => {
                        quote! {{
                            let float_mask = Tensor::<B, 2>::zeros([shape[2], shape[3]], &#mask_arg.device());
                            float_mask.mask_fill(#mask_arg.bool_not(), f32::NEG_INFINITY)
                        }}
                    }
                    _ => panic!("Unsupported mask dtype"),
                },
                _ => panic!("Attention mask must be a tensor"),
            };

            quote! {
                let shape = #attn_mask_shape;
                let #qk = #qk + #mask.expand::<4, _>(shape);
            }
        } else {
            quote! {}
        };

        // Handle causal masking
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

        // Handle qk_matmul_output at different stages
        let capped = quote! { capped };
        let (mut qk_matmul_a, mut qk_matmul_b, mut qk_matmul_c, mut qk_matmul_d) =
            (quote! {}, quote! {}, quote! {}, quote! {});
        if let Some(out_arg) = self.outputs.get(3) {
            let out = arg_to_ident(out_arg);
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

        // Handle softcap
        let softcap = if self.config.softcap != 0.0 {
            let softcap = self.config.softcap;
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
        } else {
            quote! {
                let #capped = #qk;
            }
        };

        if self.config.softmax_precision.is_some() {
            panic!("Attention: non-default softmax precision is not yet supported")
        }

        // Build output tuple
        let mut output_names = vec![output_y.clone()];
        if present_kv {
            output_names.push(arg_to_ident(self.outputs.get(1).unwrap()));
            output_names.push(arg_to_ident(self.outputs.get(2).unwrap()));
        }
        if self.outputs.get(3).is_some() {
            output_names.push(arg_to_ident(self.outputs.get(3).unwrap()));
        }
        let output = quote! { (#(#output_names,)*) };

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
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::attention::{AttentionConfig, AttentionNodeBuilder, AttentionQkMatmulOutputMode};

    #[test]
    fn test_attention_basic_rank4() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_rank3() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: Some(8),
            q_num_heads: Some(8),
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 3, DType::F32)
            .input_tensor("key", 3, DType::F32)
            .input_tensor("value", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 3>,
            key: Tensor<B, 3>,
            value: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let [batch_size, q_sequence_length, q_hidden_size] = q.dims();
                #[allow(clippy::identity_op)]
                let head_size = q_hidden_size / 8usize;
                let kv_sequence_length = k.dims()[1];
                #[allow(clippy::identity_op)]
                let v_head_size = v.dims()[2] / 8usize;
                let q = q
                    .reshape([batch_size, q_sequence_length, 8usize, head_size])
                    .permute([0, 2, 1, 3]);
                let k = k
                    .reshape([batch_size, kv_sequence_length, 8usize, head_size])
                    .permute([0, 2, 1, 3]);
                let v = v
                    .reshape([batch_size, kv_sequence_length, 8usize, v_head_size])
                    .permute([0, 2, 1, 3]);
                let scale = (1.0 / (head_size as f64).sqrt()).sqrt();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                let output = output
                    .permute([0, 2, 1, 3])
                    .reshape([batch_size as i32, q_sequence_length as i32, -1]);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_causal_mask() {
        let config = AttentionConfig {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
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
                let output = scores.matmul(v);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
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
                let qk = qk + mask.expand::<4, _>(shape);
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_softcap() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 50.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let capped = {
                    let score = qk * 0.02f64;
                    let score = score.tanh();
                    score * 50f64
                };
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_custom_scale() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: Some(0.125),
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let (output,) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = 0.3535533905932738f64;
                let q_scaled = q * scale;
                let k_scaled = k * scale;
                let k_transpose = k_scaled.transpose();
                let qk = q_scaled.matmul(k_transpose);
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output,)
            };
            output
        }
        ");
    }

    #[test]
    fn test_attention_with_past_present_kv() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::Matmul,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32) // slot 3
            .input_tensor("past_k", 4, DType::F32) // slot 4
            .input_tensor("past_v", 4, DType::F32) // slot 5
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
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
                let qk = qk + bias.expand::<4, _>(shape);
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output, present_k, present_v)
            };
            (output, present_k, present_v)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_matmul_plus_mask() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulPlusAttentionMask,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("mask", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            mask: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
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
                let qk = qk + mask.expand::<4, _>(shape);
                let qk_output = qk.clone();
                let capped = qk;
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_after_softcap() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulAfterSoftcap,
            scale: None,
            softcap: 30.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
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
                let qk = qk + bias.expand::<4, _>(shape);
                let capped = {
                    let score = qk * 0.03333333333333333f64;
                    let score = score.tanh();
                    score * 30f64
                };
                let qk_output = capped.clone();
                let scores = softmax(capped, 3);
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }

    #[test]
    fn test_attention_qk_output_mode_after_softmax() {
        let config = AttentionConfig {
            is_causal: false,
            kv_num_heads: None,
            q_num_heads: None,
            qk_matmul_output_mode: AttentionQkMatmulOutputMode::MatmulAfterSoftmax,
            scale: None,
            softcap: 0.0,
            softmax_precision: None,
        };
        let node = AttentionNodeBuilder::new("attn1")
            .input_tensor("query", 4, DType::F32)
            .input_tensor("key", 4, DType::F32)
            .input_tensor("value", 4, DType::F32)
            .input_tensor("bias", 4, DType::F32)
            .input_tensor("past_k", 4, DType::F32)
            .input_tensor("past_v", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .output_tensor("present_k", 4, DType::F32)
            .output_tensor("present_v", 4, DType::F32)
            .output_tensor("qk_output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            query: Tensor<B, 4>,
            key: Tensor<B, 4>,
            value: Tensor<B, 4>,
            bias: Tensor<B, 4>,
            past_k: Tensor<B, 4>,
            past_v: Tensor<B, 4>,
        ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
            let (output, present_k, present_v, qk_output) = {
                let q = query;
                let k = key;
                let v = value;
                let scale = (1.0 / (q.dims()[3] as f64).sqrt()).sqrt();
                let present_k = Tensor::cat([past_k, k].to_vec(), 2);
                let k = present_k.clone();
                let present_v = Tensor::cat([past_v, v].to_vec(), 2);
                let v = present_v.clone();
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
                let qk = qk + bias.expand::<4, _>(shape);
                let capped = qk;
                let scores = softmax(capped, 3);
                let qk_output = scores.clone();
                let output = scores.matmul(v);
                (output, present_k, present_v, qk_output)
            };
            (output, present_k, present_v, qk_output)
        }
        ");
    }
}
