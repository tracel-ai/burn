use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::clip::ClipNode {
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

        // Extract static values from ClipInput enum
        let min = match &self.config.min {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime min values are not supported in burn-import")
            }
            None => None,
        };
        let max = match &self.config.max {
            Some(onnx_ir::node::clip::ClipInput::Static(v)) => Some(*v),
            Some(onnx_ir::node::clip::ClipInput::Runtime(_)) => {
                panic!("Clip: runtime max values are not supported in burn-import")
            }
            None => None,
        };

        if let Some(min) = min {
            if let Some(max) = max {
                quote! {
                    let #output = #input.clamp(#min, #max);
                }
            } else {
                quote! {
                    let #output = #input.clamp_min(#min);
                }
            }
        } else if let Some(max) = max {
            quote! {
                let #output = #input.clamp_max(#max);
            }
        } else {
            panic!("Clip node must have at least one min or max value");
        }
    }
}
