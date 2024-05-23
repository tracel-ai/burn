use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct RandomUniformNode {
    pub high: f64,
    pub low: f64,
    pub output_ty: TensorType,
}

impl RandomUniformNode {
    pub fn new(output_ty: TensorType, high: f64, low: f64) -> Self {
        Self {
            high,
            low,
            output_ty,
        }
    }

    fn get_output_shape(&self) -> TokenStream {
        let shape_it = self
            .output_ty
            .shape
            .as_ref()
            .expect("RandomUniform output has no shape!")
            .iter();
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
            let #output = Tensor::random(#shape, #dist, self.device.deref());
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniform(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("core::ops::Deref");
        imports.register("burn::tensor::Distribution");
        imports.register("burn::prelude::Shape");
    }
}
