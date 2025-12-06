use super::prelude::*;

impl NodeCodegen for onnx_ir::size::SizeNode {
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
            let #output = #input.shape.num_elements();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::size::SizeNodeBuilder;

    #[test]
    fn test_size_forward() {
        let node = SizeNodeBuilder::new("size1")
            .input_tensor("input", 2, DType::F32)
            .output_scalar("output", DType::I64)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> i64 {
            let output = input.shape.num_elements();
            output
        }
        ");
    }
}
