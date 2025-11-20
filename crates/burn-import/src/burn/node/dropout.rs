use super::prelude::*;
impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::dropout::DropoutNode {
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
