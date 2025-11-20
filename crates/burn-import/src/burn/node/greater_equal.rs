use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::comparison::GreaterOrEqualNode {
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
                    quote! { #lhs_value.greater_equal(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.greater_equal(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).greater_equal(#rhs_value) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value.greater_equal_elem(#rhs_value) }
            }
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                // L >= R == R <= L
                quote! { #rhs_value.lower_equal_elem(#lhs_value) }
            }
            (ArgType::Shape(_), ArgType::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs_value as &[_], &*self.device).greater_equal(#rhs_value)
            },
            (ArgType::Tensor(_), ArgType::Shape(_)) => quote! {
                #lhs_value.greater_equal(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs_value as &[_], &*self.device))
            },
            (lhs, rhs) => panic!("greater_equal is not supported for {lhs:?} > {rhs:?}"),
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
    use onnx_ir::comparison::GreaterOrEqualNodeBuilder;

    #[test]
    fn test_greater_equal_forward() {
        let node = GreaterOrEqualNodeBuilder::new("ge1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = lhs.greater_equal(rhs);");
    }
}
