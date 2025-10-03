use onnx_ir::node::max_pool1d::MaxPool1dConfig;
use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct MaxPool1dNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: MaxPool1dConfig,
}

impl MaxPool1dNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: MaxPool1dConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    MaxPool1d
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MaxPool1dNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn field_type(&self) -> Option<Type> {
        Some(Type::Other(self.field.clone()))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = &self.field.name;
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.stride.to_tokens();
        let padding = self.config.padding.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let tokens = quote! {
            let #name = MaxPool1dConfig::new(#kernel_size)
                .with_stride(#strides)
                .with_padding(#padding)
                .with_dilation(#dilation)
                .init();
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
        imports.register("burn::nn::PaddingConfig1d");
        imports.register("burn::nn::pool::MaxPool1d");
        imports.register("burn::nn::pool::MaxPool1dConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::MaxPool1d(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for MaxPool1dNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::max_pool1d::max_pool1d_config(&node);
        let name = &node.name;
        Self::new(name, input, output, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;
    use onnx_ir::node::padding::PaddingConfig1d;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MaxPool1dNode::new(
            "max_pool1d",
            TensorType::new_float("input", 3),
            TensorType::new_float("output", 3),
            MaxPool1dConfig::new(3)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Valid)
                .with_dilation(1),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::PaddingConfig1d;
            use burn::nn::pool::MaxPool1d;
            use burn::nn::pool::MaxPool1dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                max_pool1d: MaxPool1d,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let max_pool1d = MaxPool1dConfig::new(3)
                        .with_stride(1)
                        .with_padding(PaddingConfig1d::Valid)
                        .with_dilation(1)
                        .init();

                    Self {
                        max_pool1d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
                    let output = self.max_pool1d.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
