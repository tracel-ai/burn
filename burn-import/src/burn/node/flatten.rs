use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct FlattenNode {
    pub input: TensorType,
    pub output: TensorType,
    pub start_dim: usize,
    pub end_dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for FlattenNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let start_dim = self.start_dim.to_tokens();
        let end_dim = self.end_dim.to_tokens();

        quote! {
            let #output = #input.flatten(#start_dim, #end_dim);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Flatten(self)
    }
}
