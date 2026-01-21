use super::prelude::*;

impl NodeCodegen for onnx_ir::node::mean::MeanNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let inputs = self.inputs.iter().map(|arg| scope.arg(arg));

        let output = arg_to_ident(self.outputs.first().unwrap());
        let inputs_len = self.inputs.len() as u32;

        quote! {
            let #output = (#(#inputs)+*) / #inputs_len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::mean::MeanNodeBuilder;

    #[test]
    fn test_mean() {
        let node = MeanNodeBuilder::new("mean1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .input_tensor("c", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = (a + b + c) / 3u32;
            output
        }
        ");
    }

    #[test]
    fn test_mean_two_inputs() {
        let node = MeanNodeBuilder::new("mean2")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = (a + b) / 2u32;
            output
        }
        ");
    }
}
