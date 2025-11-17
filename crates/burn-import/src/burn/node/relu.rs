use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ReluNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReluNode {
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
            let #output = burn::tensor::activation::relu(#input);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Relu(self)
    }
}

impl OnnxIntoNode for ReluNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Relu(n) = node else {
            panic!("Expected Relu node");
        };
        let input = match crate::burn::Type::from(n.inputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Relu expects tensor input"),
        };
        let output = match crate::burn::Type::from(n.outputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Relu expects tensor output"),
        };
        Self::new(input, output)
    }
}
