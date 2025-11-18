use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::argmax::ArgMaxNode {
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
        // NOTE: select_last_index=1 is not supported (will panic during conversion)
        let axis = self.config.axis.to_tokens();
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.tensor_use_owned(input_arg, node_position);
        let output = arg_to_ident(output_arg);

        match &output_arg.ty {
            onnx_ir::ir::ArgType::Tensor(tensor) => {
                if self.config.keepdims {
                    // keepdims=True: Burn's argmax keeps dimensions by default
                    quote! {
                        let #output = #input.argmax(#axis);
                    }
                } else {
                    // keepdims=False: use argmax followed by squeeze to remove the kept dimension
                    let output_rank = tensor.rank;
                    quote! {
                        let argmax_result = #input.argmax(#axis);
                        let #output = argmax_result.squeeze_dim::<#output_rank>(#axis);
                    }
                }
            }
            onnx_ir::ir::ArgType::Scalar(_) => {
                // 1D tensor with keepdims=false -> scalar output
                // ArgMax always outputs Int64 indices
                quote! {
                    let argmax_result = #input.argmax(#axis);
                    let #output = argmax_result.into_scalar().elem::<i64>();
                }
            }
            _ => panic!("ArgMax output must be Tensor or Scalar"),
        }
    }
}
