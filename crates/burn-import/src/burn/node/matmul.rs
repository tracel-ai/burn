use super::prelude::*;
use core::cmp::Ordering;

use onnx_ir::DType;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::matmul::MatMulNode {
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
            match t.dtype {
                DType::F64 | DType::F32 | DType::F16 | DType::BF16 | DType::Flex32 => {}
                _ => panic!("MatMul is only implemented for float tensors"),
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
