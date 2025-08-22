use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SumNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SumNode {
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
        let inputs = self
            .inputs
            .iter()
            .map(|t| scope.tensor_use_owned(t, node_position));

        let output = &self.output.name;

        quote! {
            let #output = #(#inputs)+*;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Sum(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{sum::SumNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_sum() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SumNode::new(
            vec![
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ],
            TensorType::new_float("tensor3", 4),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::prelude::*;

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let tensor3 = tensor1 + tensor2;

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
