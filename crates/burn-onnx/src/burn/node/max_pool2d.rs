use super::prelude::*;

impl NodeCodegen for onnx_ir::max_pool2d::MaxPool2dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let ceil_mode = self.config.ceil_mode;

        let padding = self.config.padding.to_tokens();

        Some(Field::new(
            self.name.clone(),
            quote! {
                MaxPool2d
            },
            quote! {
                let #name = MaxPool2dConfig::new(#kernel_size)
                    .with_strides(#strides)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_ceil_mode(#ceil_mode)
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
        imports.register("burn::nn::pool::MaxPool2d");
        imports.register("burn::nn::pool::MaxPool2dConfig");
        imports.register("burn::nn::PaddingConfig2d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::max_pool2d::{MaxPool2dConfig, MaxPool2dNode, MaxPool2dNodeBuilder};
    use onnx_ir::padding::PaddingConfig2d;

    fn create_max_pool2d_node(name: &str, ceil_mode: bool) -> MaxPool2dNode {
        let config =
            MaxPool2dConfig::new([3, 3], [1, 1], PaddingConfig2d::Valid, [1, 1], ceil_mode);

        MaxPool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_max_pool2d_node_asymmetric(name: &str) -> MaxPool2dNode {
        // Asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = MaxPool2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 2, 3, 4),
            [1, 1],
            false,
        );

        MaxPool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_max_pool2d_forward() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_forward_with_clone() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_field_init_ceil_mode_false() {
        let node = create_max_pool2d_node("pool1", false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_field_init_ceil_mode_true() {
        let node = create_max_pool2d_node("pool1", true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(true)
            .init();
        "#);
    }

    #[test]
    fn test_max_pool2d_forward_asymmetric_padding() {
        let node = create_max_pool2d_node_asymmetric("pool1");
        let code = codegen_forward_default(&node);
        // Asymmetric padding is now handled by the burn-nn module
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_max_pool2d_field_init_asymmetric_padding() {
        let node = create_max_pool2d_node_asymmetric("pool1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let pool1 = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 2, 3, 4))
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        ");
    }
}
