use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::space_to_depth::SpaceToDepthNode {
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
        let block_size = self.config.block_size;

        quote! {
            let #output = {
                let [b, c, h, w] = #input.shape().dims();
                #input
                    .reshape([b, c, h / #block_size, #block_size, w / #block_size, #block_size])
                    .permute([0, 3, 5, 1, 2, 4])
                    .reshape([b, c * #block_size * #block_size, h / #block_size, w / #block_size])
            };
        }
    }
}
