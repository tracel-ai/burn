use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::prelude::Shape;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ExpandNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ExpandNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![
            Type::Tensor(self.input.clone()),
            Type::Tensor(self.shape.clone()),
        ]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let shape = scope.tensor_use_owned(&self.shape, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.expand(Shape::new(#shape));
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Expand(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{expand::ExpandNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ExpandNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            TensorType::new_int("tensor3", 1),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor3".to_string()],
            vec!["tensor2".to_string()],
        );

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
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 1>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.expand(Shape::new(tensor2));

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
