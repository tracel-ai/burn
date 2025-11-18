use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::{ArgType, Argument};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::arithmetic::AddNode {
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
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = match &lhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(lhs_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(lhs_arg);
                quote! { #name }
            }
            ArgType::Shape(_) => {
                let name = arg_to_ident(lhs_arg);
                quote! { #name }
            }
        };

        let rhs = match &rhs_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(rhs_arg);
                quote! { #name }
            }
            ArgType::Shape(_) => {
                let name = arg_to_ident(rhs_arg);
                quote! { #name }
            }
        };

        let function = match (&lhs_arg.ty, &rhs_arg.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs.add(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.add(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).add(#rhs) }
                }
            }
            (ArgType::Tensor(_), ArgType::Scalar(_)) => quote! { #lhs.add_scalar(#rhs) },
            (ArgType::Scalar(_), ArgType::Tensor(_)) => quote! { #rhs.add_scalar(#lhs) },
            (ArgType::Scalar(_), ArgType::Scalar(_)) => quote! { #lhs + #rhs },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                        *result_item = result_item.saturating_add(*rhs_item);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Scalar(_)) => quote! {
                {
                    let mut result = #lhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#rhs as i64);
                    }
                    result
                }
            },
            (ArgType::Scalar(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #rhs;
                    for result_item in result.iter_mut() {
                        *result_item = result_item.saturating_add(#lhs as i64);
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Tensor(_)) => quote! {
                Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).add(#rhs)
            },
            (ArgType::Tensor(_), ArgType::Shape(_)) => quote! {
                #lhs.add(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
            },
        };

        quote! {
            let #output = #function;
        }
    }
}
