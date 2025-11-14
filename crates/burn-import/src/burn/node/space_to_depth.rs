use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct SpaceToDepthNode {
    pub input: TensorType,
    pub output: TensorType,
    pub block_size: usize,
}

impl SpaceToDepthNode {
    pub fn new(input: TensorType, output: TensorType, block_size: usize) -> Self {
        Self {
            input,
            output,
            block_size,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SpaceToDepthNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let block_size = self.block_size;

        quote! {
            let #output = {
                let [b, c, h, w] = #input.shape().dims();
                #input
                    .reshape([b, c, h / #block_size, #block_size, w / #block_size, #block_size])
                    .permute([0, 3, 5, 1, 2, 4])
                    .reshape([b, c * #block_size * #block_size, h / #block_size, w / #block_size])
            };
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::SpaceToDepth(self)
    }
}

impl OnnxIntoNode for SpaceToDepthNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs, config) = match node {
            onnx_ir::Node::SpaceToDepth {
                inputs,
                outputs,
                config,
                ..
            } => (inputs, outputs, config),
            _ => panic!("Expected SpaceToDepth node"),
        };
        let input = TensorType::from(inputs.first().unwrap());
        let output = TensorType::from(outputs.first().unwrap());
        Self::new(input, output, config.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SpaceToDepthNode::new(
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            2,
        ));

        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                    let output = {
                        let [b, c, h, w] = input.shape().dims();
                        input
                            .reshape([b, c, h / 2usize, 2usize, w / 2usize, 2usize])
                            .permute([0, 3, 5, 1, 2, 4])
                            .reshape([b, c * 2usize * 2usize, h / 2usize, w / 2usize])
                    };
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
