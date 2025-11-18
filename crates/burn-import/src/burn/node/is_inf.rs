use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::is_inf::IsInfNode {
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
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = match &input_arg.ty {
            ArgType::Tensor(_) => scope.tensor_use_owned(input_arg, node_position),
            ArgType::Scalar(_) => {
                let name = &input_arg.name;
                quote! { #name }
            }
            _ => panic!("Input must be a tensor or scalar"),
        };
        let output = arg_to_ident(output_arg);

        let function = match &output_arg.ty {
            ArgType::Scalar(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_infinite() },
                    (false, true) => quote! { #input.is_infinite() && #input.is_sign_positive() },
                    (true, false) => quote! { #input.is_infinite() && #input.is_sign_negative() },
                    (false, false) => quote! { false },
                }
            }
            ArgType::Tensor(_) => {
                match (self.config.detect_negative, self.config.detect_positive) {
                    (true, true) => quote! { #input.is_inf() },
                    (false, true) => {
                        quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                    }
                    (true, false) => {
                        quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                    }
                    (false, false) => quote! { #input.zeros_like().bool() },
                }
            }
            _ => panic!("IsInf only supports scalar or tensor outputs"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed - is_inf() is a tensor method
    }
}
