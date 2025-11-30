use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::is_nan::IsNaNNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);

        quote! {
            let #output = #input.is_nan();
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed - is_nan() is a tensor method
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::is_nan::IsNaNNodeBuilder;

    #[test]
    fn test_is_nan() {
        let node = IsNaNNodeBuilder::new("isnan1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::Bool)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.is_nan();
            output
        }
        ");
    }
}
