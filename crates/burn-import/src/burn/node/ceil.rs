use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct CeilNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for CeilNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.ceil();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Ceil(self)
    }
}

impl OnnxIntoNode for CeilNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = match Type::from(node.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("CeilNode expects tensor input"),
        };
        let output = match Type::from(node.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("CeilNode expects tensor output"),
        };
        Self::new(input, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{ceil::CeilNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(CeilNode::new(
            TensorType::new_float("tensor1", 1),
            TensorType::new_float("tensor2", 1),
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
                pub fn forward(&self, tensor1: Tensor<B, 1>) -> Tensor<B, 1> {
                    let tensor2 = tensor1.ceil();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
