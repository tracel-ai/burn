use super::{NodeCodegen, arg_to_ident};
use crate::burn::{BurnImports, Field, Scope};
use burn::record::PrecisionSettings;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

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

        let field_type = match rank {
            3 => quote! { AdaptiveAvgPool1d },
            4 => quote! { AdaptiveAvgPool2d },
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
        };

        Some(Field::new(self.name.clone(), field_type))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = Ident::new(&self.name, Span::call_site());
        let input = self.inputs.first().unwrap();
        let rank = match &input.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for GlobalAvgPool"),
        };

        let tokens = match rank {
            3 => quote! {
                let #name = AdaptiveAvgPool1dConfig::new(1)
                    .init();
            },
            4 => quote! {
                let #name = AdaptiveAvgPool2dConfig::new([1, 1])
                    .init();
            },
            dim => panic!("Unsupported input dim ({dim}) for GlobalAvgPoolNode"),
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
        // FIXME import depending on input rank
        imports.register("burn::nn::pool::AdaptiveAvgPool1d");
        imports.register("burn::nn::pool::AdaptiveAvgPool1dConfig");
        imports.register("burn::nn::pool::AdaptiveAvgPool2d");
        imports.register("burn::nn::pool::AdaptiveAvgPool2dConfig");
    }
}
