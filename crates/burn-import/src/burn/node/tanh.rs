use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::tanh::TanhNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #input.tanh();
        }
    }
}
