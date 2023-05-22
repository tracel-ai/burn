use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ReLUNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReLUNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = burn::tensor::activation::relu(#input);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::ReLU(self)
    }
}
