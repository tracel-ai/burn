use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SqrtNode {
    pub input: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SqrtNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = match &self.input {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("Sqrt input must be a tensor or scalar"),
        };
        let output = &self.output.name();

        quote! {
            let #output = #input.sqrt();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Sqrt(self)
    }
}

impl OnnxIntoNode for SqrtNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Sqrt(n) = node else {
            panic!("Expected Sqrt node");
        };
        let input = Type::from(n.inputs.first().unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}
