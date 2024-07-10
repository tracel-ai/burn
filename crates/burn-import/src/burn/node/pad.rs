use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Config, Debug)]
pub struct PadConfig {
    pub pads: Vec<i64>,
    pub constant_value: f64,
}

#[derive(Debug, Clone, new)]
pub struct PadNode {
    pub input: TensorType,
    pub output: TensorType,
    pub pad_config: PadConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for PadNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            println!("hello world")
            // let #output = #input.pad([#(#ranges),*]);
        }
    }
    fn into_node(self) -> Node<PS> {
        Node::Pad(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{pad::PadNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_pad() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(PadNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            vec![Some((0, 1)), Some((0, 1)), Some((0, 1)), Some((0, 1))],
        ));
        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.pad([Some((0, 1)), Some((0, 1)), Some((0, 1)), Some((0, 1))]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
