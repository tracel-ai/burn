use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::trilu::TriluConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TriluNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: TriluConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TriluNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let diagonal = self.config.diagonal.to_tokens();
        if self.config.upper {
            quote! {
                let #output = #input.triu(#diagonal);
            }
        } else {
            quote! {
                let #output = #input.tril(#diagonal);
            }
        }
    }
    fn into_node(self) -> super::Node<PS> {
        Node::Trilu(self)
    }
}

impl OnnxIntoNode for TriluNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Trilu(n) = node else {
            panic!("Expected Trilu node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.clone())
    }
}
