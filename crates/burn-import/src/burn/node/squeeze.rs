use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SqueezeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub axes: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        if self.axes.len() != 1 {
            panic!("Squeeze operation must specify exactly one axis");
        }

        let axis = &self.axes.first().unwrap().to_tokens();

        quote! {
            let #output = #input.squeeze(#axis);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Squeeze(self)
    }
}
