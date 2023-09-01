use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ConcatNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConcatNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let dim = self.dim.to_tokens();
        let inputs = self
            .inputs
            .iter()
            .map(|t| scope.tensor_use_owned(t, node_position));

        let output = &self.output.name;

        quote! {
            let #output = burn::tensor::Tensor::cat(vec![#(#inputs),*], #dim);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Concat(self)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{concat::ConcatNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_concat() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConcatNode::new(
            vec![
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ],
            TensorType::new_float("tensor3", 4),
            1,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }

                #[allow(clippy::let_and_return)]
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = burn::tensor::Tensor::cat(vec![tensor1, tensor2], 1);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
