use super::Node;
use crate::burn::{BurnImports, Scope, TensorDescription, ToTokens};
use burn::{nn::LinearConfig, tensor::DataSerialize};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

#[derive(Debug, Clone, new)]
pub struct LinearNode {
    pub name_field: Ident,
    pub input: TensorDescription,
    pub output: TensorDescription,
    pub data_weights: DataSerialize<f32>,
    pub data_bias: Option<DataSerialize<f32>>,
    pub config: LinearConfig,
}

impl Node for LinearNode {
    fn output_type(&self) -> TokenStream {
        let dim = self.output.dim.to_tokens();

        quote! {
            Tensor<B, #dim>
        }
    }

    fn output_name(&self) -> Ident {
        self.output.name.clone()
    }

    fn input_def(&self) -> TokenStream {
        let name = &self.input.name;
        let dim = self.output.dim.to_tokens();

        quote! {
            #name: Tensor<B, #dim>
        }
    }

    fn field_name(&self) -> Option<Ident> {
        Some(self.name_field.clone())
    }

    fn new_body(&self) -> TokenStream {
        let name = &self.name_field;
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
        let bias = self.config.bias;

        quote! {
            let #name = LinearConfig::new(#d_input, #d_output)
                .with_bias(#bias)
                .init_with(record.#name);
        }
    }

    fn new_field(&self) -> TokenStream {
        let name = &self.name_field;

        quote! {
            #name: Linear<B>,
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
        imports.register("burn::nn::Linear");
        imports.register("burn::nn::LinearConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{graph::Graph, node::test::assert_tokens, TensorDescription};
    use burn::tensor::Data;
    use proc_macro2::Span;

    #[test]
    fn test_codegen() {
        let mut graph = Graph::default();

        graph.register(LinearNode::new(
            Ident::new("linear", Span::call_site()),
            TensorDescription::new("input", 4),
            TensorDescription::new("output", 4),
            Data::from([2.]).serialize(),
            None,
            LinearConfig::new(128, 128),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::Linear;
            use burn::nn::LinearConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                linear: Linear<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let linear = LinearConfig::new(128, 128)
                        .with_bias(true)
                        .init_with(record.linear);

                    Self {
                        linear,
                    }
                }

                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.linear.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
