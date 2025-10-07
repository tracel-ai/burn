use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TransposeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub perm: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TransposeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let perm = self.perm.to_tokens();

        quote! {
            let #output = #input.permute(#perm);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Transpose(self)
    }
}

impl OnnxIntoNode for TransposeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = match crate::burn::Type::from(node.inputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Transpose expects tensor input"),
        };
        let output = match crate::burn::Type::from(node.outputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Transpose expects tensor output"),
        };
        let perm = onnx_ir::node::transpose::transpose_config(&node);
        Self::new(input, output, perm)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_transpose() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(TransposeNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            vec![0, 3, 1, 2],
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                    let tensor2 = tensor1.permute([0, 3, 1, 2]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
