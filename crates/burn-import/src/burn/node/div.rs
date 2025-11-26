use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::arithmetic::DivNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = scope.arg(lhs_arg);

        let rhs = scope.arg(rhs_arg);

        let function = match (&lhs_arg.ty, &rhs_arg.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs.div(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.div(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).div(#rhs) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => quote! { #lhs.div_scalar(#rhs) },
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                // Scalar / Tensor: convert scalar to tensor and divide
                quote! {
                    {
                        let scalar_tensor = Tensor::<B, 1>::from_data([#lhs.elem::<B::FloatElem>()], &*self.device);
                        scalar_tensor.div(#rhs)
                    }
                }
            }
            (ArgType::Scalar(_), ArgType::Scalar(_)) => quote! { #lhs / #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = if *rhs_item != 0 { *result_item / *rhs_item } else { *result_item };
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = if #rhs as i64 != 0 { *result_item / (#rhs as i64) } else { *result_item };
                    }
                    result
                }
            },
            (ArgType::Scalar(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = if *result_item != 0 { (#lhs as i64) / *result_item } else { (#lhs as i64) };
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).div(#rhs)
            },
            (ArgType::Tensor(_), ArgType::Shape(_)) => quote! {
                #lhs.div(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
            },
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::arithmetic::DivNodeBuilder;

    #[test]
    fn test_div_forward_tensor_tensor() {
        let node = DivNodeBuilder::new("div1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.div(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_div_forward_tensor_scalar() {
        let node = DivNodeBuilder::new("div1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2> {
            let output = lhs.div_scalar(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_div_forward_scalar_tensor() {
        let node = DivNodeBuilder::new("div1")
            .input_scalar("lhs", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: f32, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let scalar_tensor = Tensor::<
                    B,
                    1,
                >::from_data([lhs.elem::<B::FloatElem>()], &*self.device);
                scalar_tensor.div(rhs)
            };
            output
        }
        ");
    }
}
