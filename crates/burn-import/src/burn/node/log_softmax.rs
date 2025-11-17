use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct LogSoftmaxNode {
    pub input: TensorType,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LogSoftmaxNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let dim = self.dim.to_tokens();

        quote! {
            let #output = burn::tensor::activation::log_softmax(#input, #dim);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::LogSoftmax(self)
    }
}

impl OnnxIntoNode for LogSoftmaxNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::LogSoftmax(n) = node else {
            panic!("Expected LogSoftmax node");
        };
        let input = match Type::from(n.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("LogSoftmax expects tensor input"),
        };
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("LogSoftmax expects tensor output"),
        };
        let dim = n.config.axis;
        Self::new(input, output, dim)
    }
}
