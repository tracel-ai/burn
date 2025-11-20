use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct HardSigmoidNode {
    pub input: TensorType,
    pub output: TensorType,
    pub alpha: f64,
    pub beta: f64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for HardSigmoidNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let alpha = self.alpha.to_tokens();
        let beta = self.beta.to_tokens();

        quote! {
            let #output = burn::tensor::activation::hard_sigmoid(#input, #alpha, #beta);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::HardSigmoid(self)
    }
}

impl OnnxIntoNode for HardSigmoidNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::HardSigmoid(n) = node else {
            panic!("Expected HardSigmoid node");
        };
        let input = match crate::burn::Type::from(n.inputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("HardSigmoid expects tensor input"),
        };
        let output = match crate::burn::Type::from(n.outputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("HardSigmoid expects tensor output"),
        };
        Self::new(input, output, n.config.alpha, n.config.beta)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_hard_sigmoid() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(HardSigmoidNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            0.2,
            0.5,
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
                    let tensor2 = burn::tensor::activation::hard_sigmoid(tensor1, 0.2, 0.5);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
