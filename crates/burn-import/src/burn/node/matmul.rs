use super::prelude::*;
use core::cmp::Ordering;

impl NodeCodegen for onnx_ir::matmul::MatMulNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output_arg = self.outputs.first().unwrap();

        // Validate that lhs is a float tensor
        if let ArgType::Tensor(t) = &lhs_arg.ty {
            if !t.dtype.is_float() {
                panic!("MatMul is only implemented for float tensors");
            }
        } else {
            panic!("MatMul lhs must be a tensor");
        }

        let lhs = scope.arg(lhs_arg);
        let rhs = scope.arg(rhs_arg);
        let output = arg_to_ident(output_arg);

        // Get ranks from tensor types
        let lhs_rank = match &lhs_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("lhs must be a tensor"),
        };
        let rhs_rank = match &rhs_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("rhs must be a tensor"),
        };
        let output_rank = match &output_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("output must be a tensor"),
        };

        // Support broadcasting for missing dimensions
        match lhs_rank.cmp(&rhs_rank) {
            Ordering::Greater => {
                let num_unsqueezes = lhs_rank - rhs_rank;

                if rhs_rank == 1 {
                    // Matrix-vector product: expand vector to match matrix rank
                    let squeeze_dim = lhs_rank - 1;

                    // Build unsqueeze dimensions: [-1, 0, 0, ...]
                    let mut unsqueeze_dims = vec![-1isize];
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }

                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze_dims(&[#(#unsqueeze_dims),*])).squeeze_dim::<#output_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = lhs_rank;

                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze::<#target_rank>());
                    }
                }
            }
            Ordering::Less => {
                if lhs_rank == 1 {
                    // Vector-matrix product: expand vector to match matrix rank
                    let squeeze_dim = rhs_rank - 2;
                    let target_rank = rhs_rank;

                    quote! {
                        let #output = #lhs.unsqueeze::<#target_rank>().matmul(#rhs).squeeze_dim::<#output_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = rhs_rank;

                    quote! {
                        let #output = #lhs.unsqueeze::<#target_rank>().matmul(#rhs);
                    }
                }
            }
            Ordering::Equal => quote! {
                let #output = #lhs.matmul(#rhs);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::matmul::MatMulNodeBuilder;

    #[test]
    fn test_matmul_forward() {
        let node = MatMulNodeBuilder::new("matmul1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = lhs.matmul(rhs);
            output
        }
        ");
    }
}
