use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::eye_like::EyeLikeConfig;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[derive(Debug, Clone, new)]
pub struct EyeLikeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: EyeLikeConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for EyeLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let k_offset = self.config.k.to_token_stream();

        // Convert mask to appropriate type based on output tensor kind
        let conversion = match self.output.kind {
            crate::burn::TensorKind::Int => quote! { .int() },
            crate::burn::TensorKind::Float => quote! { .float() },
            crate::burn::TensorKind::Bool => quote! {},
        };

        // Use diag_mask to create the diagonal matrix, then invert it
        // diag_mask returns false on diagonal, true off-diagonal
        // EyeLike needs true on diagonal, false off-diagonal
        quote! {
            let #output = Tensor::diag_mask(#input.shape(), #k_offset, &*self.device).bool_not()#conversion;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::EyeLike(self)
    }
}

impl OnnxIntoNode for EyeLikeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::EyeLike(n) = node else {
            panic!("Expected EyeLike node");
        };
        let input = TensorType::from(n.inputs.first().unwrap());
        let output = TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output, n.config.clone())
    }
}
