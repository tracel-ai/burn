use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct LeakyReluNode {
    pub input: TensorType,
    pub output: TensorType,
    pub alpha: f64,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LeakyReluNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let alpha = self.alpha.to_tokens();

        quote! {
            let #output = burn::tensor::activation::leaky_relu(#input, #alpha);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::LeakyRelu(self)
    }
}

impl OnnxIntoNode for LeakyReluNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::LeakyRelu(n) = node else {
            panic!("Expected LeakyRelu node");
        };
        let input = match crate::burn::Type::from(n.inputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("LeakyRelu expects tensor input"),
        };
        let output = match crate::burn::Type::from(n.outputs.first().unwrap()) {
            crate::burn::Type::Tensor(t) => t,
            _ => panic!("LeakyRelu expects tensor output"),
        };
        let alpha = n.config.alpha;
        Self::new(input, output, alpha)
    }
}
