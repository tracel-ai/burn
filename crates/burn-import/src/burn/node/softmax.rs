use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::softmax::SoftmaxNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let dim = self.config.axis.to_tokens();

        quote! {
            let #output = burn::tensor::activation::softmax(#input, #dim);
        }
    }

    // No need to register imports since we use the fully qualified path
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::softmax::{SoftmaxConfig, SoftmaxNode, SoftmaxNodeBuilder};

    fn create_softmax_node(name: &str, axis: usize) -> SoftmaxNode {
        let config = SoftmaxConfig::new(axis);

        SoftmaxNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_softmax_forward_last_axis() {
        let node = create_softmax_node("softmax1", 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = burn::tensor::activation::softmax(input, 2);
            output
        }
        ");
    }

    #[test]
    fn test_softmax_forward_axis_0() {
        let node = create_softmax_node("softmax1", 0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = burn::tensor::activation::softmax(input, 0);
            output
        }
        ");
    }

    #[test]
    fn test_softmax_forward_axis_1() {
        let node = create_softmax_node("softmax1", 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = burn::tensor::activation::softmax(input, 1);
            output
        }
        ");
    }
}
