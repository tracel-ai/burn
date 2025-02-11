use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

#[derive(Config, Debug)]
pub struct TopKConfig {
    pub axis: usize,
    pub k: usize,
}

#[derive(Debug, Clone, new)]
pub struct TopKNode {
    pub input: TensorType,
    pub outputs: Vec<TensorType>,
    pub config: TopKConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TopKNode {
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
        let axis = self.config.axis.to_token_stream();
        let k = self.config.k.to_token_stream();

        let input = scope.tensor_use_owned(&self.input, node_position);
        let values_output = &self.outputs[0].name;
        let indices_output = &self.outputs[1].name;

        quote! {
            let (#values_output, #indices_output) = #input.topk_with_indices(#k, #axis);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::TopK(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{test::assert_tokens, top_k::TopKNode},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TopKConfig::new(1, 3);

        graph.register(TopKNode::new(
            TensorType::new_float("input_tensor", 4),
            vec![
                TensorType::new_float("values_tensor", 4),
                TensorType::new_int("indices_tensor", 4),
            ],
            config,
        ));

        graph.register_input_output(
            vec!["input_tensor".to_string()],
            vec!["values_tensor".to_string(), "indices_tensor".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
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
                pub fn forward(&self, input_tensor: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4, Int>) {
                    let (values_tensor, indices_tensor) = input_tensor.topk_with_indices(3usize, 1usize);
                    (values_tensor, indices_tensor)
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
