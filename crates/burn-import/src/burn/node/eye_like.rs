use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct EyeLikeNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for EyeLikeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = {
                let shape = #input.shape();
                let device = #input.device();

                // EyeLike creates an identity matrix with the same shape as input
                assert!(shape.dims.len() == 2, "EyeLike operation requires 2D input tensor");

                let rows = shape.dims[0];
                let cols = shape.dims[1];

                if rows == cols {
                    // Square matrix - use Tensor::eye directly
                    Tensor::eye(rows, &device)
                } else {
                    // Rectangular matrix - create zeros and place eye in top-left
                    let min_dim = core::cmp::min(rows, cols);
                    let eye_matrix = Tensor::eye(min_dim, &device);
                    let result = #input.zeros_like();

                    // Use slice assignment to place the identity matrix in the top-left corner
                    result.slice_assign([0..min_dim, 0..min_dim], eye_matrix)
                }
            };
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::EyeLike(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{eye_like::EyeLikeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(EyeLikeNode::new(
            TensorType::new_float("tensor1", 2),
            TensorType::new_float("tensor2", 2),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 2>) -> Tensor<B, 2> {
                    let tensor2 = {
                        let shape = tensor1.shape();
                        let device = tensor1.device();

                        // EyeLike creates an identity matrix with the same shape as input
                        assert!(shape.dims.len() == 2, "EyeLike operation requires 2D input tensor");

                        let rows = shape.dims[0];
                        let cols = shape.dims[1];

                        if rows == cols {
                            // Square matrix - use Tensor::eye directly
                            Tensor::eye(rows, &device)
                        } else {
                            // Rectangular matrix - create zeros and place eye in top-left
                            let min_dim = core::cmp::min(rows, cols);
                            let eye_matrix = Tensor::eye(min_dim, &device);
                            let result = tensor1.zeros_like();

                            // Use slice assignment to place the identity matrix in the top-left corner
                            result.slice_assign([0..min_dim, 0..min_dim], eye_matrix)
                        }
                    };
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
