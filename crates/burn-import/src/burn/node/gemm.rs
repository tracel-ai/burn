use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gemm::GemmNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
            .collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let a = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let b = scope.tensor_use_owned(self.inputs.get(1).unwrap(), node_position);
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
                    let c_tensor = scope.tensor_use_owned(c_input, node_position);
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
