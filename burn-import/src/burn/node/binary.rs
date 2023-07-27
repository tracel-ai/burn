use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use std::sync::Arc;

#[derive(Clone)]
pub enum BinaryType {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
}

impl BinaryType {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            BinaryType::Add => "add",
            BinaryType::Sub => "sub",
            BinaryType::Mul => "mul",
            BinaryType::Div => "div",
            BinaryType::Equal => "equal",
        }
    }
}

// Simple fn pointer that receive input as a token stream and return function call.
type FnPointer = Arc<dyn Fn(TokenStream, TokenStream) -> TokenStream>;

/// Node for all binary operators.
#[derive(Clone, new)]
pub struct BinaryNode {
    pub lhs: TensorType,
    pub rhs: TensorType,
    pub output: TensorType,
    pub binary_type: BinaryType,
    function: FnPointer,
}

impl std::fmt::Debug for BinaryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "BinaryNode {{ lhs: {:?}, rhs: {:?}, output: {:?}, name: {:?} }}",
                self.lhs,
                self.rhs,
                self.output,
                self.binary_type.as_str()
            )
            .as_str(),
        )
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BinaryNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.output)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(&self.lhs), Type::Tensor(&self.rhs)]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.tensor_use_owned(&self.lhs, node_position);
        let rhs = scope.tensor_use_owned(&self.rhs, node_position);
        let output = &self.output.name;
        let function = (self.function)(lhs, rhs);

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Binary(self)
    }
}

impl BinaryNode {
    pub(crate) fn add(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        let function = move |lhs, rhs| quote! { #lhs.add(#rhs) };
        Self::new(lhs, rhs, output, BinaryType::Add, Arc::new(function))
    }

    pub(crate) fn sub(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        let function = move |lhs, rhs| quote! { #lhs.sub(#rhs) };
        Self::new(lhs, rhs, output, BinaryType::Sub, Arc::new(function))
    }

    pub(crate) fn mul(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        let function = move |lhs, rhs| quote! { #lhs.mul(#rhs) };
        Self::new(lhs, rhs, output, BinaryType::Mul, Arc::new(function))
    }

    pub(crate) fn div(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        let function = move |lhs, rhs| quote! { #lhs.div(#rhs) };
        Self::new(lhs, rhs, output, BinaryType::Div, Arc::new(function))
    }

    pub(crate) fn equal(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        let function = move |lhs, rhs| quote! { #lhs.equal(#rhs) };
        Self::new(lhs, rhs, output, BinaryType::Equal, Arc::new(function))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::burn::node::tests::codegen_binary_operator;
    use crate::burn::TensorType;

    macro_rules! test_binary_operator {
        ($operator:ident) => {{
            codegen_binary_operator::<4, _>(
                BinaryNode::$operator(
                    TensorType::new_float("tensor1", 4),
                    TensorType::new_float("tensor2", 4),
                    TensorType::new_float("tensor3", 4),
                ),
                quote! {
                    let tensor3 = tensor1.$operator(tensor2);

                    tensor3
                },
            );
        }};
    }

    #[test]
    fn test_binary_codegen_add() {
        test_binary_operator!(add);
    }

    #[test]
    fn test_binary_codegen_sub() {
        test_binary_operator!(sub);
    }

    #[test]
    fn test_binary_codegen_mul() {
        test_binary_operator!(mul);
    }

    #[test]
    fn test_binary_codegen_div() {
        test_binary_operator!(div);
    }

    #[test]
    fn test_binary_codegen_equal() {
        test_binary_operator!(equal);
    }
}
