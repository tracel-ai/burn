use super::prelude::*;
use onnx_ir::ir::ArgType;
use quote::{ToTokens, quote};

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::eye_like::EyeLikeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let k_offset = self.config.k.to_token_stream();

        // Convert mask to appropriate type based on output tensor kind
        let output_ty = &self.outputs.first().unwrap().ty;
        let conversion = match output_ty {
            ArgType::Tensor(t) => match t.dtype {
                onnx_ir::ir::DType::I32 | onnx_ir::ir::DType::I64 => quote! { .int() },
                onnx_ir::ir::DType::F32 | onnx_ir::ir::DType::F64 => quote! { .float() },
                onnx_ir::ir::DType::Bool => quote! {},
                _ => quote! { .float() }, // Default to float
            },
            _ => quote! { .float() },
        };

        // Use diag_mask to create the diagonal matrix, then invert it
        // diag_mask returns false on diagonal, true off-diagonal
        // EyeLike needs true on diagonal, false off-diagonal
        quote! {
            let #output = Tensor::diag_mask(#input.shape(), #k_offset, &*self.device).bool_not()#conversion;
        }
    }
}
