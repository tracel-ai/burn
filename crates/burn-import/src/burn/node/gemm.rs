use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gemm::GemmNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let a = scope.arg(self.inputs.first().unwrap());
        let b = scope.arg(self.inputs.get(1).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let trans_a = self.config.trans_a;
        let trans_b = self.config.trans_b;

        // Apply transpose to A if trans_a is set
        let a = if trans_a != 0 {
            quote! { #a.transpose() }
        } else {
            quote! { #a }
        };

        // Apply transpose to B if trans_b is set
        let b = if trans_b != 0 {
            quote! { #b.transpose() }
        } else {
            quote! { #b }
        };

        // Compute A * B
        let product = quote! { #a.matmul(#b) };

        // Apply alpha scaling
        let scaled_product = match alpha {
            1.0 => product,
            _ => quote! { #product * #alpha },
        };

        // Handle optional C input with beta scaling
        if let Some(c_input) = self.inputs.get(2) {
            // Get C as either tensor or scalar depending on its type
            let c = match &c_input.ty {
                onnx_ir::ir::ArgType::Tensor(_) => {
                    let c_tensor = scope.arg(c_input);
                    quote! { #c_tensor.unsqueeze() }
                }
                onnx_ir::ir::ArgType::Scalar(_) => {
                    let c_scalar = arg_to_ident(c_input);
                    quote! { #c_scalar }
                }
                _ => panic!("C input should be Tensor or Scalar!"),
            };

            // Apply beta scaling to C
            let scaled_c = match beta {
                1.0 => c,
                _ => quote! { (#c) * #beta },
            };

            quote! {
                let #output = #scaled_product + #scaled_c;
            }
        } else {
            // No C input, just return scaled A * B
            quote! {
                let #output = #scaled_product;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gemm::{GemmConfig, GemmNode, GemmNodeBuilder};

    fn create_gemm_node_ab(
        name: &str,
        alpha: f32,
        beta: f32,
        trans_a: i64,
        trans_b: i64,
        has_c: bool,
    ) -> GemmNode {
        let config = GemmConfig::new(alpha, beta, trans_a, trans_b);
        let mut builder = GemmNodeBuilder::new(name)
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32);

        if has_c {
            builder = builder.input_tensor("c", 2, DType::F32);
        }

        builder
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_gemm_basic_ab() {
        let node = create_gemm_node_ab("gemm1", 1.0, 1.0, 0, 0, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.matmul(b);
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_alpha() {
        let node = create_gemm_node_ab("gemm1", 2.5, 1.0, 0, 0, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.matmul(b) * 2.5f32;
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_alpha_and_c() {
        let node = create_gemm_node_ab("gemm1", 2.5, 1.0, 0, 0, true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = a.matmul(b) * 2.5f32 + c.unsqueeze();
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_alpha_beta_c() {
        let node = create_gemm_node_ab("gemm1", 2.0, 3.0, 0, 0, true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = a.matmul(b) * 2f32 + (c.unsqueeze()) * 3f32;
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_trans_a() {
        let node = create_gemm_node_ab("gemm1", 1.0, 1.0, 1, 0, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.transpose().matmul(b);
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_trans_b() {
        let node = create_gemm_node_ab("gemm1", 1.0, 1.0, 0, 1, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.matmul(b.transpose());
            output
        }
        ");
    }

    #[test]
    fn test_gemm_with_trans_a_and_b() {
        let node = create_gemm_node_ab("gemm1", 1.0, 1.0, 1, 1, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.transpose().matmul(b.transpose());
            output
        }
        ");
    }
}
