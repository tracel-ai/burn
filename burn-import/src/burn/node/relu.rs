use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone, new)]
pub struct ReLUNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl Serialize for ReLUNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_none()
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReLUNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.use_owned_tensor(&self.input.name, node_position);
        let output = &self.output.name;

        quote! {
            let #output = burn::tensor::activation::relu(#input);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::ReLU(self)
    }
}
