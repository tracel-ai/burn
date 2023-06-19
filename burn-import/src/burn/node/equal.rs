use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct EqualNode {
    pub lhs: TensorType,
    pub rhs: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for EqualNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.lhs), Type::Tensor(&self.rhs)]
    }
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn field_type(&self) -> Option<Type> {
        None
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.tensor_use_owned(&self.lhs, node_position);
        let rhs = scope.tensor_use_owned(&self.rhs, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #lhs.equal(#rhs);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Equal(self)
    }
}
