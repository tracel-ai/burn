use proc_macro2::TokenStream;
use quote::quote;

use burn::record::PrecisionSettings;

use super::{Node, NodeCodegen};

use crate::burn::{Scope, TensorType, Type};

#[derive(Debug, Clone, new)]
pub struct SigmoidNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SigmoidNode {
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
            let #output = #input.sigmoid();
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Sigmoid(self)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::burn::{node::sigmoid::SigmoidNode, TensorType};

    use crate::burn::node::tests::codegen_unary_operator;

    #[test]
    fn test_codegen_node() {
        codegen_unary_operator::<4, _>(
            SigmoidNode::new(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ),
            quote! {
                let tensor2 = tensor1.sigmoid();

                tensor2
            },
        );
    }
}
