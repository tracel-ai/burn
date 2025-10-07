use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct BoolOrNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
}

impl BoolOrNode {
    pub fn new(lhs: Type, rhs: Type, output: Type) -> Self {
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BoolOrNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor or scalar"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor or scalar"),
        };

        let output = &self.output.name();

        let function = match (&self.lhs, &self.rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                if lhs_tensor.kind != TensorKind::Bool || rhs_tensor.kind != TensorKind::Bool {
                    panic!("or operation requires boolean tensors");
                }

                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                // Handle broadcasting for different ranks
                if lhs_rank == rhs_rank {
                    quote! { #lhs.bool_or(#rhs) }
                } else if lhs_rank > rhs_rank {
                    // Broadcast rhs to match lhs rank by adding leading dimensions
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.bool_or(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    // Broadcast lhs to match rhs rank by adding leading dimensions
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).bool_or(#rhs) }
                }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                if lhs_scalar.kind != ScalarKind::Bool || rhs_scalar.kind != ScalarKind::Bool {
                    panic!("or operation requires boolean scalars");
                }
                quote! { #lhs || #rhs }
            }
            _ => panic!("or is supported for tensor and scalar bool only"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Or(self)
    }
}

impl OnnxIntoNode for BoolOrNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_bool_or() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BoolOrNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 4)),
            Type::Tensor(TensorType::new_bool("tensor2", 4)),
            Type::Tensor(TensorType::new_bool("tensor3", 4)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 4, Bool>, tensor2: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
                    let tensor3 = tensor1.bool_or(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_bool_or_broadcast() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BoolOrNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 3)),
            Type::Tensor(TensorType::new_bool("tensor2", 2)),
            Type::Tensor(TensorType::new_bool("tensor3", 3)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 3, Bool>, tensor2: Tensor<B, 2, Bool>) -> Tensor<B, 3, Bool> {
                    let tensor3 = tensor1.bool_or(tensor2.unsqueeze_dims(&[0isize]));

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
