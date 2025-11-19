use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Field, Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::Argument;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::max_pool2d::MaxPool2dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        Some(Field::new(
            self.name.clone(),
            quote! {
                MaxPool2d
            },
        ))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let padding = self.config.padding.to_tokens();
        let dilation = self.config.dilation.to_tokens();

        let tokens = quote! {
            let #name = MaxPool2dConfig::new(#kernel_size)
                .with_strides(#strides)
                .with_padding(#padding)
                .with_dilation(#dilation)
                .init();
        };

        Some(tokens)
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
        imports.register("burn::nn::pool::MaxPool2d");
        imports.register("burn::nn::pool::MaxPool2dConfig");
        imports.register("burn::nn::PaddingConfig2d");
    }
}
