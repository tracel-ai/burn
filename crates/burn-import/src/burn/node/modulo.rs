use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ModNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: TensorType,
    pub fmod: bool, // false: use remainder (Python %), true: use fmod (C-style)
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ModNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;

        match (&self.lhs, &self.rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);

                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;

                // Handle broadcasting if ranks differ
                if lhs_rank != rhs_rank {
                    let (smaller_tensor, larger_tensor, smaller_rank, larger_rank) =
                        if lhs_rank < rhs_rank {
                            (&lhs, &rhs, lhs_rank, rhs_rank)
                        } else {
                            (&rhs, &lhs, rhs_rank, lhs_rank)
                        };

                    // Calculate dimensions to unsqueeze
                    let rank_diff = larger_rank - smaller_rank;
                    let unsqueeze_dims = (0..rank_diff)
                        .map(|i| {
                            let i = i as isize;
                            quote! { #i }
                        })
                        .collect::<Vec<_>>();

                    let mod_op = if self.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };

                    if lhs_rank < rhs_rank {
                        quote! {
                            let #output = #smaller_tensor
                                .unsqueeze_dims(&[#(#unsqueeze_dims),*])
                                .#mod_op(#larger_tensor);
                        }
                    } else {
                        quote! {
                            let #output = #larger_tensor.#mod_op(#smaller_tensor.unsqueeze_dims(&[#(#unsqueeze_dims),*]));
                        }
                    }
                } else {
                    let mod_op = if self.fmod {
                        quote! { fmod }
                    } else {
                        quote! { remainder }
                    };
                    quote! {
                        let #output = #lhs.#mod_op(#rhs);
                    }
                }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = rhs_scalar.name.clone();

                let mod_op = if self.fmod {
                    quote! { fmod_scalar }
                } else {
                    quote! { remainder_scalar }
                };

                quote! {
                    let #output = #lhs.#mod_op(#rhs);
                }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                panic!("Mod operation with scalar dividend and tensor divisor is not supported")
            }
            _ => panic!("Mod operation requires at least one tensor input"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Mod(self)
    }
}

impl OnnxIntoNode for ModNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::modulo::mod_config(&node);
        Self::new(lhs, rhs, output, config.fmod)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_mod_tensors_same_rank() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(ModNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            TensorType::new_float("output", 4),
            true, // Use fmod
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let output = tensor1.fmod(tensor2);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_mod_tensor_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(ModNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
            TensorType::new_float("output", 4),
            false, // Use remainder
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "scalar1".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    scalar1: f32
                ) -> Tensor<B, 4> {
                    let output = tensor1.remainder_scalar(scalar1);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_mod_broadcast_tensors() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        // Test broadcasting: 2D tensor % 4D tensor
        graph.register(ModNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            TensorType::new_float("output", 4),
            true, // Use fmod
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let output = tensor1.unsqueeze_dims(&[0isize, 1isize]).fmod(tensor2);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
