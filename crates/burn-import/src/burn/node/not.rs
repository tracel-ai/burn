use super::prelude::*;

impl NodeCodegen for onnx_ir::node::not::NotNode {
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
            let #output = #input.bool_not();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::not::NotNodeBuilder;

    #[test]
    fn test_not_forward() {
        let node = NotNodeBuilder::new("not1")
            .input_tensor("input", 2, DType::Bool)
            .output_tensor("output", 2, DType::Bool)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
            let output = input.bool_not();
            output
        }
        ");
    }
}
