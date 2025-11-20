use onnx_ir::node::dropout::DropoutConfig;
use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, OtherType, Scope, TensorType, ToTokens, Type};

#[derive(Debug, Clone)]
pub struct DropoutNode {
    pub field: OtherType,
    pub input: TensorType,
    pub output: TensorType,
    pub config: DropoutConfig,
}

impl DropoutNode {
    pub fn new<S: AsRef<str>>(
        name: S,
        input: TensorType,
        output: TensorType,
        config: DropoutConfig,
    ) -> Self {
        Self {
            field: OtherType::new(
                name,
                quote! {
                    Dropout
                },
            ),
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DropoutNode {
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
        let prob = match &self.config.prob {
            onnx_ir::node::dropout::DropoutInput::Static(val) => val.to_tokens(),
            onnx_ir::node::dropout::DropoutInput::Runtime(_) => {
                panic!("Runtime input is not implemented for Dropout")
            }
        };
        let tokens = quote! {
            let #name = DropoutConfig::new(#prob).init();
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
        imports.register("burn::nn::Dropout");
        imports.register("burn::nn::DropoutConfig");
    }

    fn into_node(self) -> Node<PS> {
        Node::Dropout(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for DropoutNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Dropout(n) = &node else {
            panic!("Expected Dropout node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(&n.name, input, output, n.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(DropoutNode::new(
            "dropout",
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            DropoutConfig {
                prob: onnx_ir::node::dropout::DropoutInput::Static(0.5),
            },
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::Dropout;
            use burn::nn::DropoutConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                dropout: Dropout,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,

            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let dropout = DropoutConfig::new(0.5)
                        .init();

                    Self {
                        dropout,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = self.dropout.forward(input);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
