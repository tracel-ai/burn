use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ReshapeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: Vec<usize>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReshapeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let shape_values = &self.shape.to_tokens();

        quote! {
            let #output = #input.reshape(Shape::new(#shape_values));
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Reshape(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{reshape::ReshapeNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            [4, 4, 4, 4].into(),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model <B: Backend>{}

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self { }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.reshape(Shape::new([4, 4, 4, 4]));

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
