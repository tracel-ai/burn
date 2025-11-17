use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SignNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SignNode {
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
            let #output = #input.sign();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Sign(self)
    }
}

impl OnnxIntoNode for SignNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Sign(n) = node else {
            panic!("Expected Sign node");
        };
        let input = match Type::from(n.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("SignNode expects tensor input"),
        };
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("SignNode expects tensor output"),
        };
        Self::new(input, output)
    }
}
