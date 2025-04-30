use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseAndNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseAndNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs
            .iter()
            .map(|t| {
                if t.kind != TensorKind::Int {
                    panic!("BitwiseAndNode only supports Int TensorType inputs");
                }
                Type::Tensor(t.clone())
            })
            .collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let inputs = self
            .inputs
            .iter()
            .map(|t| scope.tensor_use_owned(t, node_position))
            .collect::<Vec<_>>();
        let output = &self.output.name;

        quote! {
            let #output = #(#inputs)&*;
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitwiseAndNode only supports Int TensorType outputs");
        }
        Node::BitwiseAnd(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitwiseand::BitwiseAndNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitwise_and() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitwiseAndNode {
            inputs: vec![
                TensorType::new_float("input1", 1),
                TensorType::new_float("input2", 1),
            ],
            output: TensorType::new_float("output", 1),
        });
        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Debug, Clone)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                pub fn new(device: B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device),
                    }
                }
                pub fn forward(&self, input1: Tensor<B, 1>, input2: Tensor<B, 1>) -> Tensor<B, 1> {
                    let output = input1 & input2;
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}
