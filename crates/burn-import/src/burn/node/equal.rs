use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::comparison::EqualNode {
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
            ArgType::Scalar(_) | ArgType::Shape(_) => {
                let name = arg_to_ident(lhs);
                quote! { #name }
            }
        };

        let rhs_value = match &rhs.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(rhs, node_position),
            ArgType::Scalar(_) | ArgType::Shape(_) => {
                let name = arg_to_ident(rhs);
                quote! { #name }
            }
        };

        let function = match (&lhs.ty, &rhs.ty) {
            (ArgType::Tensor(lhs_tensor), ArgType::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                if lhs_rank == rhs_rank {
                    quote! { #lhs_value.equal(#rhs_value) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.equal(#rhs_value.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs_value.unsqueeze_dims(&[#(#dims),*]).equal(#rhs_value) }
                }
            }
            (ArgType::Scalar(_), ArgType::Scalar(_)) => quote! { #lhs_value == #rhs_value },
            (ArgType::Shape(_), ArgType::Shape(_)) => quote! {
                {
                    let mut result = #lhs_value;
                    for (result_item, rhs_item) in result.iter_mut().zip(#rhs_value.iter()) {
                        *result_item = if result_item == rhs_item { 1i64 } else { 0i64 };
                    }
                    result
                }
            },
            (ArgType::Shape(_), ArgType::Tensor(_)) => quote! {
                {
                    let shape_tensor = Tensor::<B, 1, Int>::from_data(#lhs_value.as_slice(), &*self.device);
                    shape_tensor.equal(#rhs_value)
                }
            },
            (ArgType::Tensor(_), ArgType::Shape(_)) => quote! {
                {
                    let shape_tensor = Tensor::<B, 1, Int>::from_data(#rhs_value.as_slice(), &*self.device);
                    #lhs_value.equal(shape_tensor)
                }
            },
            _ => panic!(
                "Comparison is supported for tensor to tensor, scalar to scalar, shape to shape, and shape to tensor only"
            ),
        };

        quote! {
            let #output = #function;
        }
    }
}
