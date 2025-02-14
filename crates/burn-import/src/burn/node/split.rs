use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Config, Debug)]
pub struct SplitConfig {
    pub axis: usize,
    pub split_size: Option<usize>,
    pub split_sizes: Option<Vec<usize>>,
}

#[derive(Debug, Clone, new)]
pub struct SplitNode {
    pub input: TensorType,
    pub outputs: Vec<TensorType>,
    pub config: SplitConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SplitNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let axis = self.config.axis.to_tokens();

        let outputs = self
            .outputs
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>();

        let unpack_outputs = quote! {
            let [#(#outputs),*] = split_tensors.try_into().unwrap();
        };

        if let Some(split_sizes) = &self.config.split_sizes {
            let split_sizes_tokens = split_sizes.to_tokens();
            quote! {
                let mut split_tensors = #input.split_with_sizes(#split_sizes_tokens, #axis);
                #unpack_outputs
            }
        } else {
            let split_size = &self.config.split_size.unwrap();
            let split_size_tokens = split_size.to_tokens();
            quote! {
                let mut split_tensors = #input.split(#split_size_tokens, #axis);
                #unpack_outputs
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Split(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{split::SplitNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_split() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SplitNode::new(
            TensorType::new_float("tensor1", 2),
            vec![
                TensorType::new_float("tensor2", 2),
                TensorType::new_float("tensor3", 2),
            ],
            SplitConfig {
                axis: 0,
                split_size: Some(2),
                split_sizes: None,
            },
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string(), "tensor3".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>,
                ) -> (Tensor<B, 2>, Tensor<B, 2>) {
                    let mut split_tensors = tensor1.split(2, 0);

                    let [tensor2, tensor3] = split_tensors.try_into().unwrap();
                        (tensor2, tensor3)
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
