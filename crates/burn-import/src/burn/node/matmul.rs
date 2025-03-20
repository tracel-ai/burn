use core::cmp::Ordering;

use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorKind, TensorType, ToTokens, Type};
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
                // Alternate unsqueeze(0) -> unsqueeze(-1) -> unsqueeze(0) -> ...
                let axes = (0..lhs_dim - rhs_dim)
                    .map(|i| if i % 2 == 0 { 0 } else { -1 })
                    .collect::<Vec<i64>>();
                let axes = axes.to_tokens();

                if rhs_dim == 1 {
                    // Matrix-vector product: squeeze(-1)
                    let squeeze_dim = lhs_dim - 1;
                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze_dims(&#axes)).squeeze(#squeeze_dim);
                    }
                } else {
                    quote! {
                        let #output = #lhs.matmul(#rhs.unsqueeze_dims(&#axes));
                    }
                }
            }
            Ordering::Less => {
                // Always unsqueeze(0)
                let axes = [0i64].repeat(rhs_dim - lhs_dim).to_tokens();

                if lhs_dim == 1 {
                    // Vector-matrix product: squeeze(-2)
                    let squeeze_dim = rhs_dim - 2;
                    quote! {
                        let #output = #lhs.unsqueeze_dims(&#axes).matmul(#rhs).squeeze(#squeeze_dim);
                    }
                } else {
                    quote! {
                        let #output = #lhs.unsqueeze_dims(&#axes).matmul(#rhs);
                    }
                }
            }
            Ordering::Equal => quote! {
                let #output = #lhs.matmul(#rhs);
            },
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Matmul(self)
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
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                    let tensor3 = tensor1.matmul(tensor2.unsqueeze_dims(&[0, -1, 0])).squeeze(3usize);

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
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                    let tensor3 = tensor1.unsqueeze_dims(&[0, 0, 0]).matmul(tensor2).squeeze(2usize);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
