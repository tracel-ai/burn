use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::relu::ReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        quote! {
            let #output = burn::tensor::activation::relu(#input);
        }
    }

    // No need to register imports since we use the fully qualified path
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::relu::ReluNodeBuilder;

    #[test]
    fn test_relu_forward() {
        let node = ReluNodeBuilder::new("relu1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = burn::tensor::activation::relu(input);");
    }
}
