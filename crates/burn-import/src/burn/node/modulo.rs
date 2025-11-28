use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::modulo::ModNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_arg = &self.inputs[0];
        let rhs_arg = &self.inputs[1];

        match (&lhs_arg.ty, &rhs_arg.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs = scope.arg(lhs_arg);
                let rhs = scope.arg(rhs_arg);

                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                // Handle broadcasting if ranks differ
                if lhs_rank != rhs_rank {
                    let (smaller_tensor, larger_tensor, smaller_rank, larger_rank) =
                        if lhs_rank < rhs_rank {
                            (&lhs, &rhs, lhs_rank, rhs_rank)
                        } else {
                            (&rhs, &lhs, rhs_rank, lhs_rank)
                        };

                    // Calculate dimensions to unsqueeze
                    let rank_diff = larger_rank - smaller_rank;
                    let unsqueeze_dims = (0..rank_diff)
                        .map(|i| {
                            let i = i as isize;
                            quote! { #i }
                        })
                        .collect::<Vec<_>>();

                    let mod_op = if self.config.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };

                    if lhs_rank < rhs_rank {
                        quote! {
                            let #output = #smaller_tensor
                                .unsqueeze_dims(&[#(#unsqueeze_dims),*])
                                .#mod_op(#larger_tensor);
                        }
                    } else {
                        quote! {
                            let #output = #larger_tensor.#mod_op(#smaller_tensor.unsqueeze_dims(&[#(#unsqueeze_dims),*]));
                        }
                    }
                } else {
                    let mod_op = if self.config.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };
                    quote! {
                        let #output = #lhs.#mod_op(#rhs);
                    }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                let lhs = scope.arg(lhs_arg);
                let rhs = Ident::new(&rhs_arg.name, Span::call_site());

                let mod_op = if self.config.fmod {
                    quote! { fmod_scalar }
                } else {
                    quote! { remainder_scalar }
                };

                quote! {
                    let #output = #lhs.#mod_op(#rhs);
                }
            }
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                panic!("Mod operation with scalar dividend and tensor divisor is not supported")
            }
            _ => panic!("Mod operation requires at least one tensor input"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::modulo::{ModConfig, ModNodeBuilder};

    #[test]
    fn test_modulo_remainder() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.remainder(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_fmod() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod2")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.fmod(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_tensor_scalar_remainder() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod3")
            .input_tensor("a", 2, DType::F32)
            .input_scalar("b", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: f32) -> Tensor<B, 2> {
            let output = a.remainder_scalar(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_tensor_scalar_fmod() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod4")
            .input_tensor("a", 2, DType::F32)
            .input_scalar("b", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: f32) -> Tensor<B, 2> {
            let output = a.fmod_scalar(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_broadcast_lhs_smaller_remainder() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod5")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = a.unsqueeze_dims(&[0isize]).remainder(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_broadcast_rhs_smaller_remainder() {
        let config = ModConfig::new(false);
        let node = ModNodeBuilder::new("mod6")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = a.remainder(b.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_modulo_broadcast_lhs_smaller_fmod() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod7")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = a.unsqueeze_dims(&[0isize]).fmod(b);
            output
        }
        ");
    }

    #[test]
    fn test_modulo_broadcast_rhs_smaller_fmod() {
        let config = ModConfig::new(true);
        let node = ModNodeBuilder::new("mod8")
            .input_tensor("a", 3, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = a.fmod(b.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }
}
