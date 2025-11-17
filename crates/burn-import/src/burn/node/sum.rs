use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SumNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SumNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let inputs = self
            .inputs
            .iter()
            .map(|t| scope.tensor_use_owned(t, node_position));

        let output = &self.output.name;

        quote! {
            let #output = #(#inputs)+*;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Sum(self)
    }
}

impl OnnxIntoNode for SumNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Sum(n) = node else {
            panic!("Expected Sum node");
        };
        let inputs = n.inputs.iter().map(TensorType::from).collect();
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(inputs, output)
    }
}
