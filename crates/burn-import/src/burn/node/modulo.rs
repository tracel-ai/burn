use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::modulo::ModNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_arg = &self.inputs[0];
        let rhs_arg = &self.inputs[1];

        match (&lhs_arg.ty, &rhs_arg.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs = scope.tensor_use_owned(lhs_arg, node_position);
                let rhs = scope.tensor_use_owned(rhs_arg, node_position);

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
                let lhs = scope.tensor_use_owned(lhs_arg, node_position);
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
