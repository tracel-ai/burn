use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct RandomNormalLikeNode {
    pub mean: f64,
    pub scale: f64,
    pub input: TensorType,
    pub output: TensorType,
}

impl RandomNormalLikeNode {
    // Set distribution parameters based on mean and scale
    fn get_distribution(&self) -> TokenStream {
        let mean = self.mean;
        let std_deviation = self.scale;
        quote! { Distribution::Normal(#mean, #std_deviation) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomNormalLikeNode {
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
        Node::RandomNormalLike(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

impl OnnxIntoNode for RandomNormalLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::RandomNormalLike(n) = node else {
            panic!("Expected RandomNormalLike node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(n.config.mean, n.config.scale, input, output)
    }
}
