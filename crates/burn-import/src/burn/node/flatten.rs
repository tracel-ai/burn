use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::flatten::FlattenNode {
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
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        if self.config.axis == 0 {
            quote! {
                let #output = #input.reshape::<2>([1, -1]);
            }
        } else {
            let axis = self.config.axis.to_tokens();
            quote! {
                let #output = {
                    let leading_dim = #input.shape().dims[..#axis].iter().product::<usize>() as i32;
                    #input.reshape::<2, _>([leading_dim, -1])
                };
            }
        }
    }
}
