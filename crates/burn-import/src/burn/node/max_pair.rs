use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct MaxPairNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl MaxPairNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MaxPairNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            _ => panic!("lhs must be a tensor"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            _ => panic!("rhs must be a tensor"),
        };

        let output = &self.output.name();

        quote! {
            let #output = #lhs.max_pair(#rhs);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Max(self)
    }
}

impl OnnxIntoNode for MaxPairNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Max(n) = node else {
            panic!("Expected Max node");
        };
        let lhs = Type::from(n.inputs.first().unwrap());
        let rhs = Type::from(n.inputs.get(1).unwrap());
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}
