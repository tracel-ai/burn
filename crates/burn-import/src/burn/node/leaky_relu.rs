use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::leaky_relu::LeakyReluNode {
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
        let alpha = self.config.alpha.to_tokens();

        quote! {
            let #output = burn::tensor::activation::leaky_relu(#input, #alpha);
        }
    }

    // No need to register imports since we use the fully qualified path
}
