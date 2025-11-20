use super::prelude::*;

impl<PS: PrecisionSettings> NodeCodegen<PS>
    for onnx_ir::node::global_avg_pool::GlobalAveragePoolNode
{
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Determine field type based on input dimension
        let input = self.inputs.first().unwrap();
        let rank = match &input.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for GlobalAvgPool"),
        };

        let name = Ident::new(&self.name, Span::call_site());

        let (field_type, init_tokens) = match rank {
            3 => (
                quote! { AdaptiveAvgPool1d },
                quote! {
                    let #name = AdaptiveAvgPool1dConfig::new(1)
                        .init();
                },
            ),
            4 => (
                quote! { AdaptiveAvgPool2d },
                quote! {
                    let #name = AdaptiveAvgPool2dConfig::new([1, 1])
                        .init();
                },
            ),
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
        };

        Some(Field::new(self.name.clone(), field_type, init_tokens))
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
        // FIXME import depending on input rank
        imports.register("burn::nn::pool::AdaptiveAvgPool1d");
        imports.register("burn::nn::pool::AdaptiveAvgPool1dConfig");
        imports.register("burn::nn::pool::AdaptiveAvgPool2d");
        imports.register("burn::nn::pool::AdaptiveAvgPool2dConfig");
    }
}
