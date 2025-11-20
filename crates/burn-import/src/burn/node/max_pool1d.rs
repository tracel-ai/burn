use super::prelude::*;
impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::max_pool1d::MaxPool1dNode {
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
        let dilation = self.config.dilation.to_tokens();

        Some(Field::new(
            self.name.clone(),
            quote! {
                MaxPool1d
            },
            quote! {
                let #name = MaxPool1dConfig::new(#kernel_size)
                    .with_stride(#strides)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .init();
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
        imports.register("burn::nn::pool::MaxPool1d");
        imports.register("burn::nn::pool::MaxPool1dConfig");
        imports.register("burn::nn::PaddingConfig1d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::max_pool1d::{MaxPool1dConfig, MaxPool1dNode, MaxPool1dNodeBuilder};
    use onnx_ir::padding::PaddingConfig1d;

    fn create_max_pool1d_node(name: &str) -> MaxPool1dNode {
        let config = MaxPool1dConfig::new(3, 1, 1, PaddingConfig1d::Valid);

        MaxPool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool1d_forward() {
        let node = create_max_pool1d_node("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = self.pool1.forward(input);");
    }

    #[test]
    fn test_max_pool1d_forward_with_clone() {
        let node = create_max_pool1d_node("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @"let output = self.pool1.forward(input.clone());");
    }
}
