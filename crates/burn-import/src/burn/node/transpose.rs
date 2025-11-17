use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TransposeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub perm: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TransposeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let perm = self.perm.to_tokens();

        quote! {
            let #output = #input.permute(#perm);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Transpose(self)
    }
}

impl OnnxIntoNode for TransposeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Transpose(n) = node else {
            panic!("Expected Transpose node");
        };
        let input = match crate::burn::Type::from(n.inputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Transpose expects tensor input"),
        };
        let output = match crate::burn::Type::from(n.outputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("Transpose expects tensor output"),
        };
        Self::new(input, output, n.config.perm.clone())
    }
}
