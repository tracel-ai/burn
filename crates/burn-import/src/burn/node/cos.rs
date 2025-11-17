use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct CosNode {
    pub input: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for CosNode {
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
            _ => panic!("Cos input must be a tensor or scalar"),
        };
        let output = &self.output.name();

        quote! {
            let #output = #input.cos();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Cos(self)
    }
}

impl OnnxIntoNode for CosNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Cos(n) = node else {
            panic!("Expected Cos node");
        };
        let input = Type::from(n.inputs.first().unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}
