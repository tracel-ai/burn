use super::prelude::*;

impl NodeCodegen for onnx_ir::node::arithmetic::MulNode {
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
                    quote! { #lhs.mul(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.mul(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).mul(#rhs) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => quote! { #lhs.mul_scalar(#rhs) },
            (ArgType::Scalar(_), ArgType::Tensor(_)) => quote! { #rhs.mul_scalar(#lhs) },
            (ArgType::Scalar(_), ArgType::Scalar(_)) => quote! { #lhs * #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = result_item.saturating_mul(*rhs_item);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_mul(#rhs as i64);
                    }
                    result
                }
            },
            (ArgType::Scalar(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_mul(#lhs as i64);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Tensor(tensor_type)) => {
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#lhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ).mul(#rhs)
                }
            }
            (ArgType::Tensor(tensor_type), ArgType::Shape(_)) => {
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    #lhs.mul(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
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
    use onnx_ir::node::arithmetic::MulNodeBuilder;

    #[test]
    fn test_mul_forward_tensor_tensor() {
        let node = MulNodeBuilder::new("mul1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.mul(rhs);
            output
        }
        ");
    }

    #[test]
    fn test_mul_forward_tensor_scalar() {
        let node = MulNodeBuilder::new("mul1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: f32) -> Tensor<B, 2> {
            let output = lhs.mul_scalar(rhs);
            output
        }
        ");
    }
}
