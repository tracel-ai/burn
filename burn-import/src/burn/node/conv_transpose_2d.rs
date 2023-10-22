use burn::{
    nn::conv::{ConvTranspose2dConfig, ConvTranspose2dRecord, ConvTranspose2dRecordItem},
    record::PrecisionSettings,
};
use quote::quote;

use crate::burn::{OtherType, TensorType, Type};

use super::{Node, NodeCodegen};

#[derive(new, Debug, Clone)]
pub struct ConvTranspose2dNode {
    name: String,
    input: TensorType,
    weight: Tensor,
    output: TensorType,
    config: ConvTranspose2dConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConvTranspose2dNode {
    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<crate::burn::Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn field_init(&self, _with_record: bool) -> Option<proc_macro2::TokenStream> {
        let name = &self.output.name;

        todo!()
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);

        let output = &self.output.name;

        todo!()
    }

    fn into_node(self) -> super::Node<PS> {
        Node::ConvTranspose2d(self)
    }
}
