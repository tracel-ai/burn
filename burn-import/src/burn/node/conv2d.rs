use super::Node;
use crate::burn::{BurnImports, Scope, TensorDescription, ToTokens};
use burn::{nn::conv::Conv2dConfig, tensor::DataSerialize};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

#[derive(Debug, Clone, new)]
pub struct Conv2d {
    pub name_field: Ident,
    pub input: TensorDescription,
    pub output: TensorDescription,
    pub data_weights: DataSerialize<f32>,
    pub data_bias: DataSerialize<f32>,
    pub config: Conv2dConfig,
}

impl Node for Conv2d {
    fn output_type(&self) -> TokenStream {
        quote! {
            Tensor<B, 4>
        }
    }

    fn output_name(&self) -> Ident {
        self.output.name.clone()
    }

    fn input_def(&self) -> TokenStream {
        let name = &self.input.name;
        let dim = self.output.dim;

        quote! {
            #name: Tensor<B, #dim>
        }
    }

    fn field_name(&self) -> Option<Ident> {
        Some(self.name_field.clone())
    }

    fn new_body(&self) -> TokenStream {
        let name = &self.name_field;
        let channels = self.config.channels.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let bias = self.config.bias;

        quote! {
            let #name = Conv2dConfig::new(#channels, #kernel_size)
                .with_stride(#stride)
                .with_dilation(#dilation)
                .with_groups(#groups)
                .with_bias(#bias)
                .init_with(record.#name);
        }
    }

    fn new_field(&self) -> TokenStream {
        let name = &self.name_field;

        quote! {
            #name: Conv2d<B>,
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.use_owned_tensor(&self.input.name, node_position);
        let output = &self.output.name;
        let field = &self.name_field;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }
    fn input_tensors(&self) -> Vec<Ident> {
        vec![self.input.name.clone()]
    }
    fn output_tensors(&self) -> Vec<Ident> {
        vec![self.output.name.clone()]
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::conv::Conv2d");
        imports.register("burn::nn::conv::Conv2dConfig");
    }
}
