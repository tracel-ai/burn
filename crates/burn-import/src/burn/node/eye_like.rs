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
            ArgType::Tensor(t) => match &t.dtype {
                dtype if dtype.is_int() || dtype.is_uint() => quote! { .int() },
                dtype if dtype.is_float() => quote! { .float() },
                dtype if dtype.is_bool() => quote! {},
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::eye_like::{EyeLikeConfig, EyeLikeNodeBuilder};

    #[test]
    fn test_eye_like_float() {
        let config = EyeLikeConfig::new(None, 0);
        let node = EyeLikeNodeBuilder::new("eye1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = Tensor::diag_mask(input.shape(), 0i64, &*self.device)
                .bool_not()
                .float();
        ");
    }

    #[test]
    fn test_eye_like_int() {
        let config = EyeLikeConfig::new(None, 1);
        let node = EyeLikeNodeBuilder::new("eye2")
            .input_tensor("input", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = Tensor::diag_mask(input.shape(), 1i64, &*self.device).bool_not().int();");
    }

    #[test]
    fn test_eye_like_bool() {
        let config = EyeLikeConfig::new(None, 0);
        let node = EyeLikeNodeBuilder::new("eye3")
            .input_tensor("input", 2, DType::Bool)
            .output_tensor("output", 2, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = Tensor::diag_mask(input.shape(), 0i64, &*self.device).bool_not();");
    }
}
