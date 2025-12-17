use super::prelude::*;

impl NodeCodegen for onnx_ir::comparison::LessNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = self.inputs.first().unwrap();
        let rhs = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_value = scope.arg(lhs);

        let rhs_value = scope.arg(rhs);

        let function = match (&lhs.ty, &rhs.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs_value.lower(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.lower(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).lower(#rhs_value) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value.lower_elem(#rhs_value) }
            }
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                // L < R == R > L
                quote! { #rhs_value.greater_elem(#lhs_value) }
            }
            (ArgType::Shape(_), ArgType::Tensor(tensor_type)) => {
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#lhs_value as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ).lower(#rhs_value)
                }
            }
            (ArgType::Tensor(tensor_type), ArgType::Shape(_)) => {
                let dtype_tokens = tensor_type.dtype.to_tokens();
                quote! {
                    #lhs_value.lower(Tensor::<B, 1, burn::tensor::Int>::from_data_dtype(
                        burn::tensor::TensorData::from(&#rhs_value as &[i64]),
                        &*self.device,
                        #dtype_tokens
                    ))
                }
            }
            (lhs, rhs) => panic!("lower is not supported for {lhs:?} > {rhs:?}"),
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
    use onnx_ir::comparison::LessNodeBuilder;

    #[test]
    fn test_less_forward() {
        let node = LessNodeBuilder::new("less1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = lhs.lower(rhs);
            output
        }
        ");
    }
}
