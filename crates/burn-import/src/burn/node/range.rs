use super::{NodeCodegen, arg_to_ident};
use crate::burn::Scope;
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use onnx_ir::ir::ArgType;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::range::RangeNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
            .collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Generate values for start, limit, and delta based on Static or Runtime
        let start = match &self.config.start {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        let limit = match &self.config.limit {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        let delta = match &self.config.delta {
            onnx_ir::node::range::RangeInput::Static(value) => {
                let literal = Literal::i64_suffixed(*value);
                quote! { #literal }
            }
            onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                match &arg.ty {
                    ArgType::Scalar(_) => {
                        let name = arg_to_ident(arg);
                        quote! { #name }
                    }
                    _ => panic!("Range parameter must be a scalar"),
                }
            }
        };

        quote! {
            let #output = Tensor::arange_step(#start..#limit, #delta as usize, &*self.device);
        }
    }
}
