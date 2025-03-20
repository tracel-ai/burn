use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SliceNode {
    pub input: TensorType,
    pub output: TensorType,
    pub ranges: Vec<Option<(i64, i64)>>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let ranges = self.ranges.iter().map(|range| match range {
            Some((start, end)) => {
                let start = start.to_tokens();
                let end = end.to_tokens();

                quote! { Some((#start, #end))}
            }
            None => quote! { None },
        });

        quote! {
            let #output = #input.slice([#(#ranges),*]);
        }
    }
    fn into_node(self) -> Node<PS> {
        Node::Slice(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{slice::SliceNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_slice() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
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
                    let tensor2 = tensor1.slice([Some((0, 1)), Some((0, 1)), Some((0, 1)), Some((0, 1))]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
