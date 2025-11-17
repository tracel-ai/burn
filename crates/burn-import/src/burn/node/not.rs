use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct NotNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for NotNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        // Not ONNX operator is constrained to bool tensors, so no need to check the type.
        quote! {
            let #output = #input.bool_not();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Not(self)
    }
}

impl OnnxIntoNode for NotNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Not(n) = node else {
            panic!("Expected Not node");
        };
        let input = match Type::from(n.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("NotNode expects tensor input"),
        };
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("NotNode expects tensor output"),
        };
        Self::new(input, output)
    }
}
