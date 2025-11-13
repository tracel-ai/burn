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
        let inputs = node.inputs().iter().map(Type::from).collect();
        let output = Type::from(node.outputs().first().unwrap());
        Self::new(inputs, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitwiseand::BitwiseAndNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitwise_and() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitwiseAndNode {
            inputs: vec![
                Type::Tensor(TensorType::new_int("input1", 1)),
                Type::Tensor(TensorType::new_int("input2", 1)),
            ],
            output: Type::Tensor(TensorType::new_int("output", 1)),
        });
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
                    let output = input1.bitwise_and(input2);
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}
