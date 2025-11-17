use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct NonZeroNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for NonZeroNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        // Generate the appropriate zero value based on input tensor type
        let zero_value = match self.input.kind {
            TensorKind::Float => quote! { 0.0 },
            TensorKind::Int => quote! { 0 },
            TensorKind::Bool => {
                // For bool tensors, we can use argwhere directly since false is the "zero" value
                // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
                // So we need to transpose the result
                return quote! {
                    let #output = #input.argwhere().transpose();
                };
            }
        };

        // For numeric tensors, create boolean mask and then get indices
        // ONNX NonZero expects output shape [rank, num_nonzero] but argwhere returns [num_nonzero, rank]
        // So we need to transpose the result
        quote! {
            let #output = #input.not_equal_elem(#zero_value).argwhere().transpose();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::NonZero(self)
    }
}

impl OnnxIntoNode for NonZeroNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::NonZero(n) = node else {
            panic!("Expected NonZero node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}
