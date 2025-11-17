use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseAndNode {
    pub inputs: Vec<Type>,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseAndNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs.clone()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        let operation = match (&self.inputs[0], &self.inputs[1]) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                quote! { #lhs.bitwise_and(#rhs) }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = &rhs_scalar.name;
                quote! { #lhs.bitwise_and_scalar(#rhs.elem()) }
            }
            (Type::Scalar(lhs_scalar), Type::Tensor(rhs_tensor)) => {
                let lhs = &lhs_scalar.name;
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                // Bitwise AND is commutative, so we can swap the order
                quote! { #rhs.bitwise_and_scalar(#lhs.elem()) }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                let lhs = &lhs_scalar.name;
                let rhs = &rhs_scalar.name;
                quote! { #lhs & #rhs }
            }
            _ => panic!("BitwiseAndNode only supports tensor and scalar inputs"),
        };

        quote! {
            let #output = #operation;
        }
    }

    fn into_node(self) -> Node<PS> {
        match &self.output {
            Type::Tensor(tensor) => {
                if tensor.kind != TensorKind::Int {
                    panic!("BitwiseAndNode only supports Int tensor outputs");
                }
            }
            Type::Scalar(scalar) => {
                if !matches!(
                    scalar.kind,
                    crate::burn::ScalarKind::Int32 | crate::burn::ScalarKind::Int64
                ) {
                    panic!("BitwiseAndNode only supports Int scalar outputs");
                }
            }
            _ => panic!("BitwiseAndNode only supports tensor and scalar outputs"),
        }
        Node::BitwiseAnd(self)
    }
}

impl OnnxIntoNode for BitwiseAndNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::BitwiseAnd(n) = node else {
            panic!("Expected BitwiseAnd node");
        };
        let inputs = n.inputs.iter().map(Type::from).collect();
        let output = Type::from(n.outputs.first().unwrap());
        Self::new(inputs, output)
    }
}
