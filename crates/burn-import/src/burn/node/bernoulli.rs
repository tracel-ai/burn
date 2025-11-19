use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use onnx_ir::ir::ArgType;
use proc_macro2::TokenStream;
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::bernoulli::BernoulliNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Use Default distribution for Bernoulli
        let dist = quote! { Distribution::Default };

        // Generate random values and compare with input to get binary output
        let input_random = quote! { #input.random_like(#dist).lower(#input) };

        // Convert to the output type based on the output tensor kind
        let output_ty = &self.outputs.first().unwrap().ty;
        let output_random = match output_ty {
            ArgType::Tensor(t) => match t.dtype {
                onnx_ir::ir::DType::Bool => input_random,
                onnx_ir::ir::DType::I32 | onnx_ir::ir::DType::I64 => quote! { #input_random.int() },
                onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => {
                    quote! { #input_random.float() }
                }
                _ => input_random, // Fallback
            },
            _ => input_random,
        };

        quote! {
            let #output = #output_random;
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}
