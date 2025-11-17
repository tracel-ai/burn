use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarType, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SizeNode {
    pub input: TensorType,
    pub output: ScalarType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SizeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Scalar(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.shape.num_elements();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Size(self)
    }
}

impl OnnxIntoNode for SizeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Size(n) = node else {
            panic!("Expected Size node");
        };
        let input = match Type::from(n.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("Size expects tensor input"),
        };
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Scalar(s) => s,
            _ => panic!("Size expects scalar output"),
        };
        Self::new(input, output)
    }
}
