use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct MeanNode {
    pub inputs: Vec<TensorType>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MeanNode {
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
        let inputs_len = self.inputs.len() as u32;

        quote! {
            let #output = (#(#inputs)+*) / #inputs_len;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Mean(self)
    }
}

impl OnnxIntoNode for MeanNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Mean(n) = node else {
            panic!("Expected Mean node");
        };
        let inputs = n.inputs.iter().map(TensorType::from).collect();
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(inputs, output)
    }
}
