use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseNotNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseNotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![{
            if self.input.kind != TensorKind::Int {
                panic!("BitwiseNotNode only supports Int TensorType inputs");
            }
            Type::Tensor(self.input.clone())
        }]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.bitwise_not();
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitwiseNotNode only supports Int TensorType outputs");
        }
        Node::BitwiseNot(self)
    }
}

impl OnnxIntoNode for BitwiseNotNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::BitwiseNot(n) = node else {
            panic!("Expected BitwiseNot node");
        };
        let input = crate::burn::TensorType::from(n.inputs.first().unwrap());
        let output = crate::burn::TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}
