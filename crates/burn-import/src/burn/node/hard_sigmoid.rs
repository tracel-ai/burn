use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::hard_sigmoid::HardSigmoidNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let alpha = self.config.alpha.to_tokens();
        let beta = self.config.beta.to_tokens();

        quote! {
            let #output = burn::tensor::activation::hard_sigmoid(#input, #alpha, #beta);
        }
    }

    // No need to register imports since we use the fully qualified path
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::hard_sigmoid::{HardSigmoidConfig, HardSigmoidNodeBuilder};

    #[test]
    fn test_hard_sigmoid() {
        let config = HardSigmoidConfig::new(0.2, 0.5);
        let node = HardSigmoidNodeBuilder::new("hsig1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = burn::tensor::activation::hard_sigmoid(input, 0.2, 0.5);
            output
        }
        ");
    }
}
