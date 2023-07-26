use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct LogSoftmaxNode {
    pub input: TensorType,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for LogSoftmaxNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let dim = self.dim.to_tokens();

        quote! {
            let #output = burn::tensor::activation::log_softmax(#input, #dim);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::LogSoftmax(self)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::burn::{node::log_softmax::LogSoftmaxNode, TensorType};

    use crate::burn::node::tests::codegen_unary_operator;

    #[test]
    fn test_codegen_node() {
        codegen_unary_operator::<4, _>(
            LogSoftmaxNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
                1,
            ),
            quote! {
                let tensor2 = burn::tensor::activation::log_softmax(tensor1, 1);

                tensor2
            },
        );
    }
}
