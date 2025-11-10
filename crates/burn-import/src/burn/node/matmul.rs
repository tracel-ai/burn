use core::cmp::Ordering;

use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct MatmulNode {
    pub lhs: TensorType,
    pub rhs: TensorType,
    pub output: TensorType,
}

impl MatmulNode {
    pub fn new(lhs: TensorType, rhs: TensorType, output: TensorType) -> Self {
        if lhs.kind != TensorKind::Float {
            panic!("MatMul is only implemented for float tensors");
        }
        Self { lhs, rhs, output }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MatmulNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![
            Type::Tensor(self.lhs.clone()),
            Type::Tensor(self.rhs.clone()),
        ]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.tensor_use_owned(&self.lhs, node_position);
        let rhs = scope.tensor_use_owned(&self.rhs, node_position);
        let output = &self.output.name;

        let lhs_dim = self.lhs.rank;
        let rhs_dim = self.rhs.rank;

        // Support broadcasting for missing dimensions
        match lhs_dim.cmp(&rhs_dim) {
            Ordering::Greater => {
                let num_unsqueezes = lhs_dim - rhs_dim;

                if rhs_dim == 1 {
                    // Matrix-vector product: expand vector to match matrix rank
                    let squeeze_dim = lhs_dim - 1;
                    let output_rank = self.output.rank;

                    // Build unsqueeze dimensions: [-1, 0, 0, ...]
                    let mut unsqueeze_dims = vec![-1isize];
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }

                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze_dims(&[#(#unsqueeze_dims),*])).squeeze_dim::<#output_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = lhs_dim;

                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze::<#target_rank>());
                    }
                }
            }
            Ordering::Less => {
                if lhs_dim == 1 {
                    // Vector-matrix product: expand vector to match matrix rank
                    let squeeze_dim = rhs_dim - 2;
                    let output_rank = self.output.rank;
                    let target_rank = rhs_dim;

                    quote! {
                        let #output = #lhs.unsqueeze::<#target_rank>().matmul(#rhs).squeeze_dim::<#output_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = rhs_dim;

                    quote! {
                        let #output = #lhs.unsqueeze::<#target_rank>().matmul(#rhs);
                    }
                }
            }
            Ordering::Equal => quote! {
                let #output = #lhs.matmul(#rhs);
            },
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::MatMul(self)
    }
}

impl OnnxIntoNode for MatmulNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = crate::burn::TensorType::from(node.inputs.first().unwrap());
        let rhs = crate::burn::TensorType::from(node.inputs.get(1).unwrap());
        let output = crate::burn::TensorType::from(node.outputs.first().unwrap());
        Self::new(lhs, rhs, output)
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{matmul::MatmulNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_matmul() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            TensorType::new_float("tensor3", 4),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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

            impl<B: Backend> Model <B> {
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
                    let tensor3 = tensor1.matmul(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_matmul_matrix_vector() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 1),
            TensorType::new_float("tensor3", 3),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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

            impl<B: Backend> Model <B> {
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
                    tensor2: Tensor<B, 1>
                ) -> Tensor<B, 3> {
                    let tensor3 = tensor1.matmul(tensor2.unsqueeze_dims(&[-1isize, 0isize, 0isize])).squeeze_dim::<3usize>(3usize);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_matmul_vector_matrix() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new_float("tensor1", 1),
            TensorType::new_float("tensor2", 4),
            TensorType::new_float("tensor3", 3),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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

            impl<B: Backend> Model <B> {
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
                    tensor1: Tensor<B, 1>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 3> {
                    let tensor3 = tensor1.unsqueeze::<4usize>().matmul(tensor2).squeeze_dim::<3usize>(2usize);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
