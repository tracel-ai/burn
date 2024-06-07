use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, TensorType, ToTokens, Type};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct ResizeNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub output_size: [i64; 2],
    pub config: InterpolateOptions,
}

impl ResizeNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        output_size: [i64; 2],
        config: InterpolateOptions,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    B
                },
            ),
            input,
            output,
            output_size,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ResizeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;

        let mode = match self.config.mode {
            InterpolateMode::Bilinear => quote! { InterpolateMode::Bilinear },
            InterpolateMode::Nearest => quote! { InterpolateMode::Nearest },
            InterpolateMode::Bicubic => quote! { InterpolateMode::Bicubic },
        };

        let tokens = quote! {
            let #name = InterpolateOptions {
                mode: #mode,
            };
        };

        Some(tokens)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let output_size = &self.output_size.to_tokens();
        let field = &self.field.name;

        quote! {
            let #output = interpolate(
                #input,
                #output_size,
                #field,
            );
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Resize(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{resize::ResizeNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ResizeNode::new(
            "resize",
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            [2, 2],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = resize(
                        tensor1,
                        [2, 2],
                        InterpolateOptions {
                            mode: InterpolateMode::Bilinear,
                        },
                    );

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
