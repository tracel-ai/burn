use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, new)]
pub struct BitShiftNode {
    pub inputs: Vec<Type>,
    pub output: Type,
    pub direction: Direction,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitShiftNode {
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
                match self.direction {
                    Direction::Left => quote! { #lhs.bitwise_left_shift(#rhs) },
                    Direction::Right => quote! { #lhs.bitwise_right_shift(#rhs) },
                }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = &rhs_scalar.name;
                match self.direction {
                    Direction::Left => quote! { #lhs.bitwise_left_shift_scalar(#rhs.elem()) },
                    Direction::Right => quote! { #lhs.bitwise_right_shift_scalar(#rhs.elem()) },
                }
            }
            (Type::Scalar(lhs_scalar), Type::Tensor(rhs_tensor)) => {
                let lhs = &lhs_scalar.name;
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                // For scalar op tensor, we need to broadcast the scalar to a tensor first
                let shift_op = match self.direction {
                    Direction::Left => quote! { _scalar_tensor.bitwise_left_shift(#rhs) },
                    Direction::Right => quote! { _scalar_tensor.bitwise_right_shift(#rhs) },
                };
                quote! {
                    {
                        let _scalar_tensor = Tensor::full(#rhs.shape(), #lhs, &#rhs.device());
                        #shift_op
                    }
                }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                let lhs = &lhs_scalar.name;
                let rhs = &rhs_scalar.name;
                match self.direction {
                    Direction::Left => quote! { #lhs << #rhs },
                    Direction::Right => quote! { #lhs >> #rhs },
                }
            }
            _ => panic!("BitShiftNode only supports tensor and scalar inputs"),
        };

        quote! {
            let #output = #operation;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::BitShift(self)
    }
}

impl OnnxIntoNode for BitShiftNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::BitShift(n) = node else {
            panic!("Expected BitShift node");
        };
        let inputs = n.inputs.iter().map(Type::from).collect();
        let output = Type::from(n.outputs.first().unwrap());
        let direction = match n.config.direction {
            onnx_ir::node::bitshift::Direction::Left => Direction::Left,
            onnx_ir::node::bitshift::Direction::Right => Direction::Right,
        };
        Self::new(inputs, output, direction)
    }
}
