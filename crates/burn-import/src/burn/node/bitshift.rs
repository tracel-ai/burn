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
        let inputs = node.inputs().iter().map(Type::from).collect();
        let output = Type::from(node.outputs().first().unwrap());
        let config = match &node {
            onnx_ir::ir::Node::BitShift { config, .. } => config,
            _ => panic!("Expected BitShift node"),
        };
        let direction = match config.direction {
            onnx_ir::node::bitshift::Direction::Left => Direction::Left,
            onnx_ir::node::bitshift::Direction::Right => Direction::Right,
        };
        Self::new(inputs, output, direction)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitshift::BitShiftNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitshift_left() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitShiftNode::new(
            vec![
                Type::Tensor(TensorType::new_int("input1", 1)),
                Type::Tensor(TensorType::new_int("input2", 1)),
            ],
            Type::Tensor(TensorType::new_int("output", 1)),
            Direction::Left,
        ));

        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input1: Tensor<B, 1, Int>, input2: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
                    let output = input1.bitwise_left_shift(input2);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_bitshift_right() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitShiftNode::new(
            vec![
                Type::Tensor(TensorType::new_int("input1", 1)),
                Type::Tensor(TensorType::new_int("input2", 1)),
            ],
            Type::Tensor(TensorType::new_int("output", 1)),
            Direction::Right,
        ));

        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
            &[],
            &[],
        );

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input1: Tensor<B, 1, Int>, input2: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
                    let output = input1.bitwise_right_shift(input2);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
