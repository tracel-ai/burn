use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, TensorType, Type};
use burn::module::Module;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Module, Debug, Clone)]
pub enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(new, Module, Debug, Clone)]
pub struct ResizeOptions {
    pub mode: ResizeMode,
}

#[derive(Debug, Clone)]
pub struct ResizeNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub output_size: TensorType,
    pub config: ResizeOptions,
}

impl ResizeNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        output_size: TensorType,
        config: ResizeOptions,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    burn::module::Ignored<InterpolateOptions>
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
        vec![
            Type::Tensor(self.input.clone()),
            Type::Tensor(self.output_size.clone()),
        ]
    }

    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;

        let mode = match self.config.mode {
            ResizeMode::Linear => quote! { InterpolateMode::Bilinear },
            ResizeMode::Nearest => quote! { InterpolateMode::Nearest },
            ResizeMode::Cubic => quote! { InterpolateMode::Bicubic },
        };

        let tokens = quote! {
            let #name = InterpolateOptions {
                mode: #mode,
            };
            let #name = burn::module::Ignored(#name);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output_size = scope.tensor_use_owned(&self.output_size, node_position);
        let output = &self.output.name;

        let field = &self.field.name;

        quote! {
            let output_size_raw = #output_size.to_data().value;
            let mut output_size = [0usize; 2];

            for (i, &x) in output_size_raw.iter().rev().take(2).rev().enumerate() {
                output_size[i] = x.elem::<i64>() as usize;
            }

            let #output = interpolate(
                #input,
                output_size,
                self.#field.0.clone(),
            );
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Resize(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::ElementConversion");
        imports.register("burn::tensor::module::interpolate");
        imports.register("burn::tensor::ops::InterpolateMode");
        imports.register("burn::tensor::ops::InterpolateOptions");
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
            TensorType::new_int("output_size", 1),
            ResizeOptions::new(ResizeMode::Linear),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "output_size".to_string()],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::module::interpolate;
            use burn::tensor::ops::InterpolateMode;
            use burn::tensor::ops::InterpolateOptions;
            use burn::tensor::ElementConversion;
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                resize: burn::module::Ignored<InterpolateOptions>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let resize = InterpolateOptions {
                        mode: InterpolateMode::Bilinear,
                    };
                    let resize = burn::module::Ignored(resize);
                    Self {
                        resize,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    output_size: Tensor<B, 1, Int>
                ) -> Tensor<B, 4> {
                    let output_size_raw = output_size.to_data().value;
                    let mut output_size = [0usize; 2];

                    for (i, &x) in output_size_raw.iter().rev().take(2).rev().enumerate() {
                        output_size[i] = x.elem::<i64>() as usize;
                    }

                    let tensor2 = interpolate(tensor1, output_size, self.resize.0.clone());

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
