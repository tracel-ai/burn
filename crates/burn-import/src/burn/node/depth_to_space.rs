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
            DepthToSpaceMode::Dcr => {
                quote! {
                    let [b, c, h, w] = #input.shape().dims();
                    #input
                        .reshape([b, #block_size, #block_size, c / (#block_size * #block_size), h, w])
                        .permute([0, 3, 4, 1, 5, 2])
                        .reshape([b, c / (#block_size * #block_size), h * #block_size, w * #block_size])
                }
            }
            DepthToSpaceMode::Crd => {
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
        let onnx_ir::Node::DepthToSpace(n) = node else {
            panic!("Expected DepthToSpace node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config)
    }
}
