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
        let onnx_ir::Node::SpaceToDepth(n) = node else {
            panic!("Expected SpaceToDepth node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.block_size)
    }
}
