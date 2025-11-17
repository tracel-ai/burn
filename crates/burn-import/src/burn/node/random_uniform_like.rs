use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomUniformLikeNode {
    pub low: f64,
    pub high: f64,
    pub input: TensorType,
    pub output: TensorType,
}

impl RandomUniformLikeNode {
    // Set distribution parameters based on low and high
    fn get_distribution(&self) -> TokenStream {
        let low = self.low;
        let high = self.high;
        quote! { Distribution::Uniform(#low, #high) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomUniformLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;
        let input = scope.tensor_use_owned(&self.input, node_position);
        let dist = self.get_distribution();
        quote! {
            let #output = #input.random_like(#dist);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniformLike(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

impl OnnxIntoNode for RandomUniformLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::RandomUniformLike(n) = node else {
            panic!("Expected RandomUniformLike node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(n.config.low, n.config.high, input, output)
    }
}
