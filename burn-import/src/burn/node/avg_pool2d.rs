use proc_macro2::TokenStream;
use quote::quote;

use burn::{nn::pool::AvgPool2dConfig, record::PrecisionSettings};

use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct AvgPool2dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: AvgPool2dConfig,
}

impl AvgPool2dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: AvgPool2dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    AvgPool2d
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for AvgPool2dNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self, _with_record: bool) -> Option<TokenStream> {
        let name = &self.field.name;
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let padding = self.config.padding.to_tokens();

        let init_line = quote! {
            init();
        };

        let tokens = quote! {
            let #name = AvgPool2dConfig::new(#kernel_size)
                .with_strides(#strides)
                .with_padding(#padding)
                .#init_line
        };

        Some(tokens)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let field = &self.field.name;

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::pool::AvgPool2d");
        imports.register("burn::nn::pool::AvgPool2dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::AvgPool2d(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{avg_pool2d::AvgPool2dNode, test::assert_tokens},
        TensorType,
    };
    use burn::{nn::pool::AvgPool2dConfig, nn::PaddingConfig2d, record::FullPrecisionSettings};

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(AvgPool2dNode::new(
            "avg_pool2d",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            AvgPool2dConfig::new([3, 3])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Valid),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::PaddingConfig2d;
            use burn::nn::pool::AvgPool2d;
            use burn::nn::pool::AvgPool2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                avg_pool2d: AvgPool2d,
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let avg_pool2d = AvgPool2dConfig::new([3, 3])
                        .with_strides([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init();

                    Self {
                        avg_pool2d,
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.avg_pool2d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
