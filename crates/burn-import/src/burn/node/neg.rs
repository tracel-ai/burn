use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::neg::NegNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs.iter().collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        use onnx_ir::ir::ArgType;

        let input_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let input = match &input_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(input_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(input_arg);
                quote! { #name }
            }
            _ => panic!("Neg only supports tensor or scalar inputs"),
        };

        let neg_expr = match &input_arg.ty {
            ArgType::Tensor(_) => quote! { #input.neg() },
            ArgType::Scalar(_) => quote! { -#input },
            _ => panic!("Neg only supports tensor or scalar inputs"),
        };

        quote! {
            let #output = #neg_expr;
        }
    }
}
