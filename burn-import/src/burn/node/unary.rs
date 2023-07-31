use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use std::sync::Arc;

// Simple fn pointer that receive input as a token stream and return function call.
type FnPointer = Arc<dyn Fn(TokenStream) -> TokenStream>;

/// Node for all unary operators.
#[derive(Clone, new)]
pub struct UnaryNode {
    pub input: TensorType,
    pub output: TensorType,
    pub kind: UnaryNodeKind,
    function: FnPointer,
}

/// Type of unary node.
#[derive(Clone)]
pub enum UnaryNodeKind {
    Flatten,
    Relu,
    Sigmoid,
    LogSoftmax,
    Transpose,
}

impl UnaryNodeKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Flatten => "flatten",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::LogSoftmax => "log_softmax",
            Self::Transpose => "transpose",
        }
    }
}

impl std::fmt::Debug for UnaryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "UnaryNode {{ input: {:?}, output: {:?}, name: {} }}",
                self.input,
                self.output,
                self.kind.as_str()
            )
            .as_str(),
        )
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnaryNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.input)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let function = (self.function)(input);

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unary(self)
    }
}

impl UnaryNode {
    pub(crate) fn flatten(
        input: TensorType,
        output: TensorType,
        start_dim: usize,
        end_dim: usize,
    ) -> Self {
        let start_dim = start_dim.to_tokens();
        let end_dim = end_dim.to_tokens();
        let function = move |input| quote! { #input.flatten(#start_dim, #end_dim) };

        Self::new(input, output, UnaryNodeKind::Flatten, Arc::new(function))
    }

    pub(crate) fn relu(input: TensorType, output: TensorType) -> Self {
        let function = move |input| quote! { burn::tensor::activation::relu(#input) };
        Self::new(input, output, UnaryNodeKind::Relu, Arc::new(function))
    }

    pub(crate) fn sigmoid(input: TensorType, output: TensorType) -> Self {
        let function = move |input| quote! { burn::tensor::activation::sigmoid(#input) };
        Self::new(input, output, UnaryNodeKind::Sigmoid, Arc::new(function))
    }

    pub(crate) fn log_softmax(input: TensorType, output: TensorType, dim: usize) -> Self {
        let dim = dim.to_tokens();
        let function = move |input| quote! { burn::tensor::activation::log_softmax(#input, #dim) };
        Self::new(input, output, UnaryNodeKind::LogSoftmax, Arc::new(function))
    }

    pub(crate) fn transpose(input: TensorType, output: TensorType) -> Self {
        let function = move |input| quote! { #input.transpose() };
        Self::new(input, output, UnaryNodeKind::Transpose, Arc::new(function))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::node::tests::codegen_unary_operator;
    use crate::burn::TensorType;

    #[test]
    fn test_unary_codegen_flatten() {
        codegen_unary_operator::<4, _>(
            UnaryNode::flatten(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
                1,
                2,
            ),
            quote! {
                let tensor2 = tensor1.flatten(1, 2);

                tensor2
            },
        );
    }

    #[test]
    fn test_unary_codegen_relu() {
        codegen_unary_operator::<4, _>(
            UnaryNode::relu(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ),
            quote! {
                let tensor2 = burn::tensor::activation::relu(tensor1);

                tensor2
            },
        );
    }

    #[test]
    fn test_unary_codegen_sigmoid() {
        codegen_unary_operator::<4, _>(
            UnaryNode::sigmoid(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ),
            quote! {
                let tensor2 = burn::tensor::activation::sigmoid(tensor1);

                tensor2
            },
        );
    }

    #[test]
    fn test_unary_codegen_log_softmax() {
        codegen_unary_operator::<4, _>(
            UnaryNode::log_softmax(
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

    #[test]
    fn test_unary_codegen_transpose() {
        codegen_unary_operator::<4, _>(
            UnaryNode::transpose(
                TensorType::new_float("tensor1", 4),
                TensorType::new_float("tensor2", 4),
            ),
            quote! {
                let tensor2 = tensor1.transpose();

                tensor2
            },
        );
    }
}
