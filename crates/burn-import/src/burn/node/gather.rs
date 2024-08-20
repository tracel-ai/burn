use super::{Node, NodeCodegen};
use crate::burn::{TensorType, ToTokens, Type};

use burn::record::PrecisionSettings;
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
        let input_rank = match &self.input {
            Type::Tensor(in_tensor) => in_tensor.dim,
            Type::Shape(_) => 1,
            _ => panic!("Gather needs Tensor or Shape input, got {:?}!", self.input),
        };
        let index_rank = match &self.index {
            Type::Tensor(idx_tensor) => idx_tensor.dim,
            Type::Scalar(_) => 0, 
            _ => panic!("Gather needs Scalar or Tensor index, got {:?}!", self.index),
        };
        let output_rank = index_rank + input_rank - 1;
        
        let input = match &self.input {
            Type::Tensor(in_tensor) => scope.tensor_use_owned(in_tensor, node_position),
            Type::Shape(in_shape) => {
                let in_shape_name = &in_shape.name;
                // To copy just the values from the shape value without moving it
                // (which could lead to ownership problems if the same Shape is used multiple times)
                // borrow the array as a slice and use that to create the Tensor:
                quote! { Tensor::<B, 1, Int>::from_data(&#in_shape_name as &[_], &*self.device) }
            }
            _ => panic!("Gather needs Scalar or Shape input, got {:?}!", self.input),
        };
        let output = &self.output.name;
        let output_kind = match &self.output.kind {
            crate::burn::TensorKind::Int => quote! { Int },
            crate::burn::TensorKind::Float => quote! { Float },
            crate::burn::TensorKind::Bool => quote! { Bool },
        };
        let kind_import = match &self.output.kind {
            crate::burn::TensorKind::Int => quote! { use burn::tensor::{Int, Shape}; },
            crate::burn::TensorKind::Float => quote! { use burn::tensor::{Float, Shape}; },
            crate::burn::TensorKind::Bool => quote! { use burn::tensor::{Bool, Shape}; },
        };
        let out_final = match &self.index {
            Type::Scalar(_) => quote! {
                let #output = Tensor::cat(out, #dim).squeeze::<#output_rank>(#dim);
            },
            _ => quote! {
                let #output = Tensor::cat(out, #dim);
            }
        };

        let gather = quote! {
            extern crate alloc;
            use alloc::vec::Vec;
            #kind_import
            let mut out = Vec::new();

            let n_dims = indices.dims().len();
            let index_flat = match n_dims {
                nd if nd == 1 => indices.reshape([1, -1]),
                nd if nd >= 2 => indices.flatten::<2>(0, nd - 2),
                _ => panic!("Number of dimensions must be greater than 0"),
            };

            for idxs in index_flat.iter_dim(0) {
                let idxs = idxs.squeeze::<1>(0);
                let slice = Tensor::select(
                    #input.clone(),
                    #dim,
                    idxs,
                );
                let slice_shape = Tensor::shape(&slice);
                let mut shape: Vec<usize> = slice_shape.clone().into();
                shape.insert(#dim, 1);
                let reshaped: Tensor::<B, #output_rank, #output_kind> = slice.reshape(Shape::from(shape));
                out.push(reshaped);
            }
            #out_final
        };

        match &self.index {
            Type::Scalar(idx_scalar) => {
                // To do a scalar select (select just a single index in one dim),
                // convert the 0-D index to a 1-D Tensor with len 1 to use burn's select,
                // then squeeze the dimension to reduce the rank
                let index = &idx_scalar.name;
                quote! {
                    let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                    #gather
                }
            }
            Type::Tensor(idx_tensor) => {
                let index = scope.tensor_use_owned(idx_tensor, node_position);
                quote! {
                    let indices = #index;
                    #gather
                }
            }
            _ => panic!("Gather needs Scalar or Tensor index, got {:?}!", self.index),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }
}




#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

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
                    let indices = tensor2;
                    extern crate alloc;
                    use alloc::vec::Vec;
                    use burn::tensor::{Float, Shape};
                    let mut out = Vec::new();

                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        nd if nd == 1 => indices.reshape([1, -1]),
                        nd if nd >= 2 => indices.flatten::<2>(0, nd - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    for idxs in index_flat.iter_dim(0) {
                        let idxs = idxs.squeeze::<1>(0);
                        let slice = Tensor::select(
                            tensor1.clone(),
                            0,
                            idxs,
                        );
                        let slice_shape = Tensor::shape(&slice);
                        let mut shape: Vec<usize> = slice_shape.clone().into();
                        shape.insert(0, 1);
                        let reshaped: Tensor::<B, 2usize, Float> = slice.reshape(Shape::from(shape));
                        out.push(reshaped);
                    }
                    let tensor3 = Tensor::cat(out, 0);
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
            TensorType::new_int("tensor2", 1),
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
                ) -> Tensor<B, 1, Int> {
                    let indices = tensor1;
                    extern crate alloc;
                    use alloc::vec::Vec;
                    use burn::tensor::{Int, Shape};
                    let mut out = Vec::new();
                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        nd if nd == 1 => indices.reshape([1, -1]),
                        nd if nd >= 2 => indices.flatten::<2>(0, nd - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    for idxs in index_flat.iter_dim(0) {
                        let idxs = idxs.squeeze::<1>(0);
                        let slice = Tensor::select(
                            Tensor::<B, 1, Int>::from_data(&shape1 as &[_], &*self.device).clone(),
                            0,
                            idxs,
                        );
                        let slice_shape = Tensor::shape(&slice);
                        let mut shape: Vec<usize> = slice_shape.clone().into();
                        shape.insert(0, 1);
                        let reshaped: Tensor::<B, 1usize, Int> = slice.reshape(Shape::from(shape));
                        out.push(reshaped);
                    }
                    let tensor2 = Tensor::cat(out, 0);

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
            TensorType::new_float("tensor2", 1),
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
                ) -> Tensor<B, 1> {
                    let indices = Tensor::<B, 1, _>::from_data([scalar1], &*self.device);
                    extern crate alloc;
                    use alloc::vec::Vec;
                    use burn::tensor::{Float, Shape};
                    let mut out = Vec::new();

                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        nd if nd == 1 => indices.reshape([1, -1]),
                        nd if nd >= 2 => indices.flatten::<2>(0, nd - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    for idxs in index_flat.iter_dim(0) {
                        let idxs = idxs.squeeze::<1>(0);
                        let slice = Tensor::select(
                            tensor1.clone(),
                            0,
                            idxs,
                        );
                        let slice_shape = Tensor::shape(&slice);
                        let mut shape: Vec<usize> = slice_shape.clone().into();
                        shape.insert(0, 1);
                        let reshaped: Tensor::<B, 1usize, Float> = slice.reshape(Shape::from(shape));
                        out.push(reshaped);
                    }
                    let tensor2 = Tensor::cat(out, 0).squeeze::<1usize>(0);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
