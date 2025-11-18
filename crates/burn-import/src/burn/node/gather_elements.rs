use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens};

use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gather_elements::GatherElementsNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
            .collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> proc_macro2::TokenStream {
        let dim = self.config.axis.to_tokens();
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let index = scope.tensor_use_owned(&self.inputs[1], node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = #input.gather(#dim, #index);
        }
    }
}
