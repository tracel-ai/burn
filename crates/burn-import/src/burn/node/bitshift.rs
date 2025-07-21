use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, Type};
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
    pub output: TensorType,
    pub direction: Direction,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitShiftNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs.clone()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;

        let operation = match (&self.inputs[0], &self.inputs[1], self.direction) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor), Direction::Left) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                quote! { #lhs.bitwise_left_shift(#rhs) }
            }
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor), Direction::Right) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                quote! { #lhs.bitwise_right_shift(#rhs) }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar), Direction::Left) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = &rhs_scalar.name;
                quote! { #lhs.bitwise_left_shift_scalar(#rhs.elem()) }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar), Direction::Right) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = &rhs_scalar.name;
                quote! { #lhs.bitwise_right_shift_scalar(#rhs.elem()) }
            }
            (Type::Scalar(lhs_scalar), Type::Tensor(rhs_tensor), Direction::Left) => {
                let lhs = &lhs_scalar.name;
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                // For scalar << tensor, we need to broadcast the scalar to a tensor first
                quote! {
                    {
                        let _scalar_tensor = Tensor::full(#rhs.shape(), #lhs, &#rhs.device());
                        _scalar_tensor.bitwise_left_shift(#rhs)
                    }
                }
            }
            (Type::Scalar(lhs_scalar), Type::Tensor(rhs_tensor), Direction::Right) => {
                let lhs = &lhs_scalar.name;
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                // For scalar >> tensor, we need to broadcast the scalar to a tensor first
                quote! {
                    {
                        let _scalar_tensor = Tensor::full(#rhs.shape(), #lhs, &#rhs.device());
                        _scalar_tensor.bitwise_right_shift(#rhs)
                    }
                }
            }
            (Type::Scalar(_), Type::Scalar(_), _) => {
                panic!("BitShiftNode does not support both inputs as scalars")
            }
            _ => panic!("BitShiftNode only supports tensor and scalar inputs"),
        };

        quote! {
            let #output = #operation;
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitShiftNode only supports Int TensorType outputs");
        }
        Node::BitShift(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register ElementConversion for scalar operations
        for input in &self.inputs {
            if matches!(input, Type::Scalar(_)) {
                imports.register("burn::tensor::ElementConversion");
                break;
            }
        }
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
            TensorType::new_int("output", 1),
            Direction::Left,
        ));

        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
            TensorType::new_int("output", 1),
            Direction::Right,
        ));

        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
