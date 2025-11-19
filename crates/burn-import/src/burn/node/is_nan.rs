use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::is_nan::IsNaNNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = match &input_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(input_arg, node_position),
            ArgType::Scalar(_) => {
                let name = arg_to_ident(input_arg);
                quote! { #name }
            }
            _ => panic!("Input must be a tensor or scalar"),
        };
        let output = arg_to_ident(output_arg);

        quote! {
            let #output = #input.is_nan();
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed - is_nan() is a tensor method
    }
}
