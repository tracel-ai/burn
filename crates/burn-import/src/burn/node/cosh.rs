use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct CoshNode {
    pub input: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for CoshNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = match &self.input {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("Cosh input must be a tensor or scalar"),
        };
        let output = &self.output.name();

        quote! {
            let #output = #input.cosh();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Cosh(self)
    }
}

impl OnnxIntoNode for CoshNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        Self::new(input, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_cosh() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(CoshNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.cosh();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
