use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct CeilNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for CeilNode {
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
            let #output = #input.ceil();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Ceil(self)
    }
}

impl OnnxIntoNode for CeilNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Ceil(n) = node else {
            panic!("Expected Ceil node");
        };
        let input = match Type::from(n.inputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("CeilNode expects tensor input"),
        };
        let output = match Type::from(n.outputs.first().unwrap()) {
            Type::Tensor(t) => t,
            _ => panic!("CeilNode expects tensor output"),
        };
        Self::new(input, output)
    }
}
