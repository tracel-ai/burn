use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::depth_to_space::{DepthToSpaceConfig, DepthToSpaceMode};
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct DepthToSpaceNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: DepthToSpaceConfig,
}

impl DepthToSpaceNode {
    pub fn new(input: TensorType, output: TensorType, config: DepthToSpaceConfig) -> Self {
        Self {
            input,
            output,
            config,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DepthToSpaceNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let block_size = self.config.block_size;

        let output_expr = match self.config.mode {
            DepthToSpaceMode::DCR => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, #block_size, #block_size, c / (#block_size * #block_size), h, w])
                        .permute([0, 3, 4, 1, 5, 2])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
            DepthToSpaceMode::CRD => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, c / (#block_size * #block_size), #block_size, #block_size, h, w])
                        .permute([0, 1, 4, 2, 5, 3])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
        };
        quote! {
            let #output = {
                #output_expr
            };
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::DepthToSpace(self)
    }
}

impl OnnxIntoNode for DepthToSpaceNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::depth_to_space::DepthToSpaceConfig>();
        Self::new(input, output, config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_dcr() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(DepthToSpaceNode::new(
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            DepthToSpaceConfig::new(DepthToSpaceMode::DCR, 2),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

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
                            .reshape([b, 2usize, 2usize, c / (2usize * 2usize), h, w])
                            .permute([0, 3, 4, 1, 5, 2])
                            .reshape([b, c / (2usize * 2usize), h * 2usize, w * 2usize])
                    };
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_crd() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(DepthToSpaceNode::new(
            TensorType::new_float("input", 4),
            TensorType::new_float("output", 4),
            DepthToSpaceConfig::new(DepthToSpaceMode::CRD, 2),
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

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
                            .reshape([b, c / (2usize * 2usize), 2usize, 2usize, h, w])
                            .permute([0, 1, 4, 2, 5, 3])
                            .reshape([b, c / (2usize * 2usize), h * 2usize, w * 2usize])
                    };
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
