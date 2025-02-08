use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub input: TensorType,
    pub output: TensorType,
    pub num_classes: usize,
    pub on_value: f32,
    pub off_value: f32,
    pub axis: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let num_classes = &self.num_classes;
        let on_value = &self.on_value;
        let off_value = &self.off_value;
        let axis = &self.axis;

        quote! {
            let #output = #input.one_hot_fill(#num_classes, #on_value, #off_value, #axis);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::OneHot(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{one_hot::OneHotNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(OneHotNode::new(
            TensorType::new_float("tensor1", 1),
            TensorType::new_float("tensor2", 2),
            3,
            1.0,
            0.0,
            -1,
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

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 1>) -> Tensor<B, 2> {
                    let tensor2 = tensor1.one_hot_fill(3usize, 1f32, 0f32, -1i64);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
