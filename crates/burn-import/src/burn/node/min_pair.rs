use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct MinPairNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl MinPairNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MinPairNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            _ => panic!("lhs must be a tensor"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            _ => panic!("rhs must be a tensor"),
        };

        let output = &self.output.name();

        quote! {
            let #output = #lhs.min_pair(#rhs);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Min(self)
    }
}

impl OnnxIntoNode for MinPairNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_min_pair() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MinPairNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            Type::Tensor(TensorType::new_float("tensor3", 4)),
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

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.min_pair(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
