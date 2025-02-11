use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub input: TensorType,
    pub output: TensorType,
    pub num_classes: usize,
    pub values: [f32; 2],
    pub values_type: TensorType,
    pub axis: i64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        let mut new_output = self.output.clone();
        new_output.kind = self.values_type.kind;
        vec![Type::Tensor(new_output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let num_classes = &self.num_classes;
        let on_value = &self.values[1];
        let off_value = &self.values[0];
        let axis = &self.axis;
        let input_type = &self.input.kind;
        let output_type = &self.values_type.kind; // output is tied to values type
        match (input_type, output_type) {
            (TensorKind::Int, TensorKind::Int) | (TensorKind::Float, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value.into(), #off_value.into(), #axis);
                }
            }
            (TensorKind::Int, TensorKind::Float) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value.into(), #off_value.into(), #axis).float();
                }
            }
            (TensorKind::Float, TensorKind::Int) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value.into(), #off_value.into(), #axis).int();
                }
            }
            (TensorKind::Int, TensorKind::Bool) | (TensorKind::Float, TensorKind::Bool) => {
                quote! {
                    let #output = #input.one_hot_fill(#num_classes, #on_value.into(), #off_value.into(), #axis).bool();
                }
            }
            (TensorKind::Bool, _) => panic!("Input should be numeric"),
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
            [0., 1.],
            TensorType::new_float("tensor3", 1),
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
                    let tensor2 = tensor1
                        .one_hot_fill(3usize, 1f32.into(), 0f32.into(), -1i64);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
