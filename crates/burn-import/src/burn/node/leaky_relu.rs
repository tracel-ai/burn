use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::leaky_relu::LeakyReluNode {
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

        quote! {
            let #output = burn::tensor::activation::leaky_relu(#input, #alpha);
        }
    }

    // No need to register imports since we use the fully qualified path
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::leaky_relu::{LeakyReluConfig, LeakyReluNode, LeakyReluNodeBuilder};

    fn create_leaky_relu_node(name: &str, alpha: f64) -> LeakyReluNode {
        let config = LeakyReluConfig::new(alpha);

        LeakyReluNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_leaky_relu_forward_default_alpha() {
        let node = create_leaky_relu_node("leaky_relu1", 0.01);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = burn::tensor::activation::leaky_relu(input, 0.01);");
    }

    #[test]
    fn test_leaky_relu_forward_custom_alpha() {
        let node = create_leaky_relu_node("leaky_relu1", 0.2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = burn::tensor::activation::leaky_relu(input, 0.2);");
    }
}
