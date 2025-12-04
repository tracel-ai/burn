use super::prelude::*;
impl NodeCodegen for onnx_ir::dropout::DropoutNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let prob = match &self.config.prob {
            onnx_ir::dropout::DropoutInput::Static(val) => val.to_tokens(),
            onnx_ir::dropout::DropoutInput::Runtime(_) => {
                panic!("Runtime input is not implemented for Dropout")
            }
        };

        Some(Field::new(
            self.name.clone(),
            quote! {
                Dropout
            },
            quote! {
                let #name = DropoutConfig::new(#prob).init();
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::Dropout");
        imports.register("burn::nn::DropoutConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dropout::{DropoutConfig, DropoutInput, DropoutNodeBuilder};

    #[test]
    fn test_dropout_forward() {
        let config = DropoutConfig::new(DropoutInput::Static(0.5));
        let node = DropoutNodeBuilder::new("dropout1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = self.dropout1.forward(input);
            output
        }
        ");
    }
}
