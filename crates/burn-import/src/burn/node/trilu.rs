use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::trilu::TriluNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let diagonal = self.config.diagonal.to_tokens();

        if self.config.upper {
            quote! {
                let #output = #input.triu(#diagonal);
            }
        } else {
            quote! {
                let #output = #input.tril(#diagonal);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::trilu::{TriluConfig, TriluNodeBuilder};

    #[test]
    fn test_trilu_upper() {
        let config = TriluConfig::new(true, 0);
        let node = TriluNodeBuilder::new("triu1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.triu(0);
            output
        }
        ");
    }

    #[test]
    fn test_trilu_lower() {
        let config = TriluConfig::new(false, 1);
        let node = TriluNodeBuilder::new("tril1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.tril(1);
            output
        }
        ");
    }
}
