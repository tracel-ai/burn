use super::prelude::*;
impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::avg_pool1d::AveragePool1dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.stride.to_tokens();
        let padding = self.config.padding.to_tokens();
        let count_include_pad = self.config.count_include_pad;

        Some(Field::new(
            self.name.clone(),
            quote! {
                AvgPool1d
            },
            quote! {
                let #name = AvgPool1dConfig::new(#kernel_size)
                    .with_stride(#strides)
                    .with_padding(#padding)
                    .with_count_include_pad(#count_include_pad)
                    .init();
            },
        ))
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::pool::AvgPool1d");
        imports.register("burn::nn::pool::AvgPool1dConfig");
        imports.register("burn::nn::PaddingConfig1d");
    }
}
