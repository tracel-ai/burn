use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::matmulinteger::MatMulIntegerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = scope.arg(self.inputs.first().unwrap());
        let rhs = scope.arg(self.inputs.get(1).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Get ranks for handling broadcasting
        let lhs_rank = match &self.inputs.first().unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for lhs"),
        };
        let rhs_rank = match &self.inputs.get(1).unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for rhs"),
        };

        // Handle zero-points: synthesize when missing, otherwise lift to input rank
        let lhs_zp = if let Some(zp_input) = self.inputs.get(2) {
            let zp = scope.arg(zp_input);
            if lhs_rank > 1 {
                quote! { (#zp).unsqueeze::<#lhs_rank>() }
            } else {
                quote! { #zp }
            }
        } else {
            quote! { Tensor::zeros_like(&#lhs) }
        };

        let rhs_zp = if let Some(zp_input) = self.inputs.get(3) {
            let zp = scope.arg(zp_input);
            if rhs_rank > 1 {
                quote! { (#zp).unsqueeze::<#rhs_rank>() }
            } else {
                quote! { #zp }
            }
        } else {
            quote! { Tensor::zeros_like(&#rhs) }
        };

        // Centered inputs (subtract zero-points)
        let lhs_centered = quote! { (#lhs).sub(#lhs_zp) };
        let rhs_centered = quote! { (#rhs).sub(#rhs_zp) };

        // Handle rank differences for matmul broadcasting
        match lhs_rank.cmp(&rhs_rank) {
            std::cmp::Ordering::Greater => {
                let num_unsqueezes = lhs_rank - rhs_rank;

                if rhs_rank == 1 {
                    // Matrix-vector product: expand vector to match matrix rank
                    let squeeze_dim = lhs_rank - 1;
                    let out_rank = lhs_rank - 1;

                    // Build unsqueeze dimensions: [-1, 0, 0, ...]
                    let mut unsqueeze_dims = vec![-1isize];
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }

                    quote! {
                        let #output = (#lhs_centered).matmul((#rhs_centered).unsqueeze_dims(&[#(#unsqueeze_dims),*])).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = lhs_rank;
                    quote! {
                        let #output = (#lhs_centered).matmul((#rhs_centered).unsqueeze::<#target_rank>());
                    }
                }
            }
            std::cmp::Ordering::Less => {
                if lhs_rank == 1 {
                    // Vector-matrix product: expand vector to match matrix rank
                    let squeeze_dim = rhs_rank - 2;
                    let out_rank = rhs_rank - 1;
                    let target_rank = rhs_rank;
                    quote! {
                        let #output = (#lhs_centered).unsqueeze::<#target_rank>().matmul(#rhs_centered).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = rhs_rank;
                    quote! {
                        let #output = (#lhs_centered).unsqueeze::<#target_rank>().matmul(#rhs_centered);
                    }
                }
            }
            std::cmp::Ordering::Equal => {
                quote! {
                    let #output = (#lhs_centered).matmul(#rhs_centered);
                }
            }
        }
    }
}
