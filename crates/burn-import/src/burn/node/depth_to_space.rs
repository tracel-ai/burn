use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, depth_to_space::DepthToSpaceMode};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::depth_to_space::DepthToSpaceNode {
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

        let output_expr = match self.config.mode {
            DepthToSpaceMode::Dcr => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, #block_size, #block_size, c / (#block_size * #block_size), h, w])
                        .permute([0, 3, 4, 1, 5, 2])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
            DepthToSpaceMode::Crd => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, c / (#block_size * #block_size), #block_size, #block_size, h, w])
                        .permute([0, 1, 4, 2, 5, 3])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
        };
        quote! {
            let #output = {
                #output_expr
            };
        }
    }
}
