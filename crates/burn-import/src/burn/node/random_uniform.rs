use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct RandomUniformNode {
    pub low: f64,
    pub high: f64,
    pub output_ty: TensorType,
    pub shape: Vec<usize>,
}

impl RandomUniformNode {
    pub fn new(output_ty: TensorType, low: f64, high: f64, shape: Vec<usize>) -> Self {
        Self {
            low,
            high,
            output_ty,
            shape,
        }
    }

    fn get_output_shape(&self) -> TokenStream {
        let shape_it = self.shape.iter();
        quote! { Shape::new([#(#shape_it),*]) }
    }

    fn get_distribution(&self) -> TokenStream {
        let low = self.low;
        let high = self.high;
        quote! { Distribution::Uniform(#low, #high) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomUniformNode {
    fn input_types(&self) -> Vec<Type> {
        Vec::with_capacity(0)
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output_ty.clone())]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output_ty.name;
        let shape = self.get_output_shape();
        let dist = self.get_distribution();
        quote! {
            let #output = Tensor::random(#shape, #dist, &*self.device);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniform(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

impl OnnxIntoNode for RandomUniformNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::RandomUniform(n) = node else {
            panic!("Expected RandomUniform node");
        };
        let output_type = TensorType::from(n.outputs.first().unwrap());

        Self::new(
            output_type,
            n.config.low,
            n.config.high,
            n.config.shape.clone(),
        )
    }
}
