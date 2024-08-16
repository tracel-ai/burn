use super::{Node, NodeCodegen};
use crate::burn::{TensorType, ToTokens, Type};

use burn::{record::PrecisionSettings, tensor::{backend::Backend, Int, Tensor}};
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct GatherNode {
    pub input: Type,
    pub index: Type,
    pub output: TensorType,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![self.input.clone(), self.index.clone()]
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input = match &self.input {
            Type::Tensor(in_tensor) => scope.tensor_use_owned(in_tensor, node_position),
            Type::Shape(in_shape) => {
                let in_shape_name = &in_shape.name;
                // To copy just the values from the shape value without moving it
                // (which could lead to ownership problems if the same Shape is used multiple times)
                // borrow the array as a slice and use that to create the Tensor:
                quote! { Tensor::from_data(&#in_shape_name as &[_], &*self.device) }
            }
            _ => panic!("Gather needs Scalar or Shape input, got {:?}!", self.input),
        };
        let output = &self.output.name;

        match &self.index {
            Type::Scalar(idx_scalar) => {
                // To do a scalar select (select just a single index in one dim),
                // convert the 0-D index to a 1-D Tensor with len 1 to use burn's select,
                // then squeeze the dimension to reduce the rank
                let index = &idx_scalar.name;
                quote! {
                    //let #output = #input.select(#dim, Tensor::from_data([#index], &*self.device)).squeeze(#dim);
                    let #output = gather(#dim, #input, Tensor::from_data([#index], &*self.device)).squeeze(#dim);;
                }
            }
            Type::Tensor(idx_tensor) => {
                let index = scope.tensor_use_owned(idx_tensor, node_position);
                quote! {
                    //let #output = #input.select(#dim, #index);
                    let #output = gather(#dim, #input, #index);
                }
            }
            _ => panic!("Gather needs Scalar or Tensor index, got {:?}!", self.index),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }
}


fn gather<B, const D: usize, const X: usize, const O: usize>(
    axis: usize,
    input: Tensor<B, D>,
    index: Tensor<B, X, Int>
) -> Tensor<B, O>
where
    B: Backend,
{
    let n_dims_index = index.dims().len();
    let mut out = vec![];
    let index_flat = index.flatten::<2>(0, n_dims_index - 2);
    for idxs in index_flat.iter_dim(0) {
        let idxs = idxs.squeeze::<1>(0);
        let subtensor = input.clone().select(axis, idxs);
        out.push(subtensor);
    }
    Tensor::stack(out, axis)
}


#[cfg(test)]
mod tests {

    use burn::{
        backend::ndarray::{NdArray, NdArrayDevice},
        record::FullPrecisionSettings,
        tensor::{Int, Tensor},
    };

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{gather::GatherNode, test::assert_tokens},
        ScalarKind, ScalarType, ShapeType, TensorType,
    };

    #[test]
    fn test_codegen_gather() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_int("tensor2", 1)),
            TensorType::new_float("tensor3", 2),
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                    tensor1: Tensor<B, 2>,
                    tensor2: Tensor<B, 1, Int>
                ) -> Tensor<B, 2> {
                    //let tensor3 = tensor1.select(0, tensor2);
                    let tensor3 = gather(0, tensor1, tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_gather_shape_input() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Shape(ShapeType::new("shape1", 3)),
            Type::Tensor(TensorType::new_int("tensor1", 1)),
            TensorType::new_float("tensor2", 2),
            0,
        ));

        graph.register_input_output(
            vec!["shape1".to_string(), "tensor1".to_string()],
            vec!["tensor2".to_string()],
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
                    shape1: [usize; 3],
                    tensor1: Tensor<B, 1, Int>
                ) -> Tensor<B, 2> {
                    let tensor2 = Tensor::from_data(&shape1 as &[_], &*self.device).select(0, tensor1);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_gather_scalar_idx() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int64)),
            TensorType::new_float("tensor2", 2),
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "scalar1".to_string()],
            vec!["tensor2".to_string()],
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
                    tensor1: Tensor<B, 2>,
                    scalar1: i64
                ) -> Tensor<B, 2> {
                    //let tensor2 = tensor1.select(0, Tensor::from_data([scalar1], &*self.device)).squeeze(0);
                    let tensor2 = gather(0, tensor1, Tensor::from_data([scalar1], &*self.device)).squeeze(0);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn gather() {
        let device = NdArrayDevice::default();

        let test_cases = vec![
            (
                0,
                Tensor::<NdArray, 2>::from_data([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], &device),
                Tensor::<NdArray, 2, Int>::from_data([[0, 1], [1, 2]], &device),
                Tensor::<NdArray, 3>::from_data(
                    [
                     [[1.0, 1.2],
                      [2.3, 3.4]],
                     [[2.3, 3.4],
                      [4.5, 5.7]]
                    ],
                    &device
                )
            ),
            (
                1,
                Tensor::<NdArray, 2>::from_data([[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]], &device),
                Tensor::<NdArray, 2, Int>::from_data([[0, 2]], &device),
                Tensor::<NdArray, 3>::from_data(
                    [
                     [[1.0, 1.9]],
                     [[2.3, 3.9]],
                     [[4.5, 5.9]]
                    ],
                    &device
                )
            )
        ];
        for (axis, input, index, expected) in test_cases {
            let out = super::gather(axis, input, index);
            assert!(out.all_close(expected, None, None));
        }
    }
}
