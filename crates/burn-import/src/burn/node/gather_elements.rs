use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct GatherElementsNode {
    pub input: TensorType,
    pub index: TensorType,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherElementsNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![
            Type::Tensor(self.input.clone()),
            Type::Tensor(self.index.clone()),
        ]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input = scope.tensor_use_owned(&self.input, node_position);
        let index = scope.tensor_use_owned(&self.index, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.gather(#dim, #index);
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::GatherElements(self)
    }
}

impl OnnxIntoNode for GatherElementsNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::GatherElements(n) = node else {
            panic!("Expected GatherElements node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let index = TensorType::from(n.inputs.get(1).unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, index, output, n.config.axis)
    }
}
