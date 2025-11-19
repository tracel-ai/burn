use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::bitwisexor::BitwiseXorNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = self.inputs.first().unwrap();
        let rhs = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs_value = match &lhs.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(lhs, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(lhs);
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor or scalar"),
        };

        let rhs_value = match &rhs.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(rhs);
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor or scalar"),
        };

        let function = match (&lhs.ty, &rhs.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs_value.bitwise_xor(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.bitwise_xor(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).bitwise_xor(#rhs_value) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value.bitwise_xor_scalar((#rhs_value as i64).elem()) }
            }
            (ArgType::Scalar(_), ArgType::Tensor(_)) => {
                quote! { #rhs_value.bitwise_xor_scalar((#lhs_value as i64).elem()) }
            }
            (ArgType::Scalar(_), ArgType::Scalar(_)) => {
                quote! { #lhs_value ^ #rhs_value }
            }
            _ => panic!("BitwiseXor operation requires tensor or scalar inputs"),
        };

        quote! {
            let #output = #function;
        }
    }
}
