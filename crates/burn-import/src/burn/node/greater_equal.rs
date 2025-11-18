use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::comparison::GreaterOrEqualNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs.iter().collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
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
            ArgType::Shape(_) => {
                let name = arg_to_ident(lhs);
                quote! { #name }
            }
        };

        let rhs_value = match &rhs.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(rhs);
                quote! { #name }
            }
            ArgType::Shape(_) => {
                let name = arg_to_ident(rhs);
                quote! { #name }
            }
        };

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
