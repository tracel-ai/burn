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

        let gather = match &self.index {
            Type::Scalar(_) => quote! {
                let slice = Tensor::select(#input, #dim, indices);
                let #output = slice.squeeze::<#output_rank>(#dim);
            },
            _ => match index_rank {
                1 => quote! {
                    let #output = Tensor::select(#input, #dim, indices);
                },
                _ => quote! {
                    extern crate alloc;
                    use alloc::vec::Vec;
                    let mut out = Vec::new();

                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        1 => indices.reshape([1, -1]),
                        n if n >= 2 => indices.flatten::<2>(0, n - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    for idxs in index_flat.iter_dim(0) {
                        let idxs = idxs.squeeze::<1>(0);
                        let slice = Tensor::select(#input.clone(), #dim, idxs);
                        out.push(slice);
                    }
                    let #output = Tensor::stack::<#output_rank>(out, #dim);
                },
            },
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

    use burn::{record::FullPrecisionSettings, tensor::TensorData};

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{gather::GatherNode, test::assert_tokens},
        ScalarKind, ScalarType, ShapeType, TensorType,
    };

    #[test]
    fn test_codegen_gather_idx_1d() {
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
                    let tensor3 = Tensor::select(tensor1, 0, indices);
                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_gather_idx_2d() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_int("tensor2", 2)),
            TensorType::new_float("tensor3", 3),
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
                    tensor2: Tensor<B, 2, Int>
                ) -> Tensor<B, 3> {
                    let indices = tensor2;
                    extern crate alloc;
                    use alloc::vec::Vec;
                    let mut out = Vec::new();

                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        1 => indices.reshape([1, -1]),
                        n if n >= 2 => indices.flatten::<2>(0, n - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    for idxs in index_flat.iter_dim(0) {
                        let idxs = idxs.squeeze::<1>(0);
                        let slice = Tensor::select(tensor1.clone(), 0, idxs);
                        out.push(slice);
                    }
                    let tensor3 = Tensor::stack::<3usize>(out, 0);
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

                    let tensor2 = Tensor::select(
                        Tensor::<B, 1, Int>::from_data(&shape1 as &[_], &*self.device),
                        0,
                        indices,
                    );

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

                    let slice = Tensor::select(tensor1, 0, indices);
                    let tensor2 = slice.squeeze::<1usize>(0);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_gather_tensor() {
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
            pub fn forward(&self, input1: Tensor<B, 2>, input2: Tensor<B, 2, Int>) -> Tensor<B, 3> {
                let indices = input2;
                extern crate alloc;
                use alloc::vec::Vec;
                use burn::tensor::{Float, Shape};
                let mut out = Vec::new();
                let n_dims = indices.dims().len();
                let index_flat = match n_dims {
                    1 => indices.reshape([1, -1]),
                    n if n >= 2 => indices.flatten::<2>(0, n - 2),
                    _ => panic!("Number of dimensions must be greater than 0"),
                };
                for idxs in index_flat.iter_dim(0) {
                    let idxs = idxs.squeeze::<1>(0);
                    let slice = Tensor::select(input1.clone(), 0, idxs);
                    let slice_shape = Tensor::shape(&slice);
                    let mut shape: Vec<usize> = slice_shape.clone().into();
                    shape.insert(0, 1);
                    let reshaped: Tensor<B, 3usize, Float> = slice.reshape(Shape::from(shape));
                    out.push(reshaped);
                }
                let gather1_out1 = Tensor::cat(out, 0);
                gather1_out1
            }
        }

        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        type B = burn::backend::NdArray;
        let model = Model::<B>::new(&device);
        let input = Tensor::<B, 2>::from_data([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], &device);
        let index = Tensor::<B, 2, Int>::from_data([[0, 1], [1, 2]], &device);
        let expected = Tensor::<B, 3>::from([[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected.to_data());
    }

    #[test]
    fn test_gather_scalar_idx() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_int("tensor2", 1)),
            TensorType::new_float("tensor3", 1),
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

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
            pub fn forward(&self, input1: Tensor<B, 2>, input2: i64) -> Tensor<B, 1> {
                let indices = Tensor::<B, 1, _>::from_data([input2], &*self.device);
                let slice = Tensor::select(input1, 0, indices);
                let gather1_out1 = slice.squeeze::<1usize>(0);
                gather1_out1
            }
        }

        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        type B = burn::backend::NdArray;
        let model = Model::<B>::new(&device);
        let input = Tensor::<B, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = 0;
        let expected = TensorData::from([1f32, 2., 3.]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn test_gather_shape_input() {
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
            pub fn forward(
                &self,
                input1: [usize; 3],
                input2: Tensor<B, 1, Int>,
            ) -> Tensor<B, 1, Int> {
                let indices = input2;
                let gather1_out1 = Tensor::select(
                    Tensor::<B, 1, Int>::from_data(&input1 as &[_], &*self.device),
                    0,
                    indices,
                );
                gather1_out1
            }
        }

        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        type B = burn::backend::NdArray;
        let model = Model::<B>::new(&device);
        let input = [2, 3, 4];
        let index = Tensor::<B, 1, Int>::from_ints([0], &device);
        let expected = TensorData::from([2i64]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }
}
