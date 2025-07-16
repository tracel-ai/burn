use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone)]
pub struct GatherNode {
    pub input: Type,
    pub index: GatherIndices,
    pub output: Type,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub enum GatherIndices {
    /// Static indices known at compile time
    Static(Vec<i64>),
    /// Runtime indices as a Type
    Runtime(Type),
}

impl GatherNode {
    pub fn new(input: Type, index: Type, output: Type, dim: usize) -> Self {
        Self {
            input,
            index: GatherIndices::Runtime(index),
            output,
            dim,
        }
    }

    pub fn with_static_indices(input: Type, indices: Vec<i64>, output: Type, dim: usize) -> Self {
        Self {
            input,
            index: GatherIndices::Static(indices),
            output,
            dim,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        match &self.index {
            GatherIndices::Runtime(index_type) => vec![self.input.clone(), index_type.clone()],
            GatherIndices::Static(_) => vec![self.input.clone()],
        }
    }

    fn forward(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input_rank = match &self.input {
            Type::Tensor(in_tensor) => in_tensor.rank,
            Type::Shape(_) => 1,
            _ => panic!("Gather needs Tensor or Shape input, got {:?}!", self.input),
        };

        let output = &self.output.name();

        // Handle Shape input with CPU computation
        if matches!(&self.input, Type::Shape(_)) {
            let input_shape = match &self.input {
                Type::Shape(in_shape) => in_shape,
                _ => unreachable!(),
            };

            match &self.output {
                Type::Scalar(_) => {
                    // Gathering a single element from a shape produces a scalar
                    match &self.index {
                        GatherIndices::Runtime(Type::Scalar(idx_scalar)) => {
                            let index = &idx_scalar.name;
                            let input_shape_name = &input_shape.name;
                            let output = &self.output.name();
                            quote! {
                                let input_shape = &#input_shape_name;
                                let #output = input_shape[#index as usize];
                            }
                        }
                        GatherIndices::Static(indices) => {
                            if indices.len() != 1 {
                                panic!(
                                    "Static indices length {} doesn't match scalar output",
                                    indices.len()
                                );
                            }
                            let idx = indices[0] as usize;
                            let input_shape_name = &input_shape.name;
                            let output = &self.output.name();
                            quote! {
                                let input_shape = &#input_shape_name;
                                let #output = input_shape[#idx] as i64;
                            }
                        }
                        _ => panic!(
                            "Gather from Shape to Scalar needs scalar index, got {:?}!",
                            self.index
                        ),
                    }
                }
                Type::Shape(out_shape) => {
                    match &self.index {
                        GatherIndices::Runtime(Type::Tensor(idx_tensor)) => {
                            let index = scope.tensor_use_owned(idx_tensor, node_position);
                            let index_rank = idx_tensor.rank;
                            let output_rank = out_shape.rank;
                            let input_shape_name = &input_shape.name;

                            if index_rank == 1 {
                                quote! {
                                    let input_shape = &#input_shape_name;
                                    let indices_data = #index.to_data();
                                    let indices_vec: alloc::vec::Vec<usize> = indices_data.iter::<i64>().map(|x| x as usize).collect();
                                    let mut output_shape = alloc::vec::Vec::with_capacity(indices_vec.len());
                                    for &idx in &indices_vec {
                                        output_shape.push(input_shape[idx]);
                                    }
                                    let #output: [usize; #output_rank] = output_shape.try_into().unwrap();
                                }
                            } else {
                                panic!(
                                    "Multi-dimensional indices for Shape gather not yet supported"
                                );
                            }
                        }
                        GatherIndices::Static(indices) => {
                            let output_rank = out_shape.rank;
                            let input_shape_name = &input_shape.name;
                            let indices_len = indices.len();

                            if indices_len != output_rank {
                                panic!(
                                    "Static indices length {indices_len} doesn't match output rank {output_rank}"
                                );
                            }

                            // Generate static gathering code
                            let gather_elements = indices.iter().map(|&idx| {
                                let idx_usize = idx as usize;
                                quote! { input_shape[#idx_usize] }
                            });

                            quote! {
                                let input_shape = &#input_shape_name;
                                let #output: [usize; #output_rank] = [#(#gather_elements),*];
                            }
                        }
                        _ => panic!(
                            "Gather from Shape to Shape needs Tensor index, got {:?}!",
                            self.index
                        ),
                    }
                }
                _ => panic!(
                    "Gather from Shape input can only output Shape or Scalar, got {:?}!",
                    self.output
                ),
            }
        } else {
            // Handle Tensor input with tensor operations
            let input = match &self.input {
                Type::Tensor(in_tensor) => scope.tensor_use_owned(in_tensor, node_position),
                _ => unreachable!(),
            };

            match &self.output {
                Type::Tensor(_) => {
                    match &self.index {
                        GatherIndices::Runtime(Type::Scalar(idx_scalar)) => {
                            // To do a scalar select (select just a single index in one dim),
                            // convert the 0-D index to a 1-D Tensor with len 1 to use burn's select,
                            // then squeeze the dimension to reduce the rank
                            let index = &idx_scalar.name;
                            let output_rank = input_rank - 1;

                            if output_rank == 0 {
                                // If output rank is 0, squeeze and keep as 1D tensor
                                quote! {
                                    let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                                    let #output = Tensor::select(#input, #dim, indices);
                                }
                            } else {
                                quote! {
                                    let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                                    let slice = Tensor::select(#input, #dim, indices);
                                    let #output = slice.squeeze::<#output_rank>(#dim);
                                }
                            }
                        }
                        GatherIndices::Runtime(Type::Tensor(idx_tensor)) => {
                            let index = scope.tensor_use_owned(idx_tensor, node_position);
                            let index_rank = idx_tensor.rank;
                            let output_rank = index_rank + input_rank - 1;
                            let final_rank = output_rank.max(1); // Ensure minimum rank of 1

                            match index_rank {
                                1 => {
                                    quote! {
                                        let indices = #index;
                                        let #output = Tensor::select(#input, #dim, indices);
                                    }
                                }
                                _ => quote! {
                                    let indices = #index;

                                    let n_dims = indices.dims().len();
                                    let index_flat = match n_dims {
                                        1 => indices.reshape([1, -1]),
                                        n if n >= 2 => indices.flatten::<2>(0, n - 2),
                                        _ => panic!("Number of dimensions must be greater than 0"),
                                    };

                                    let out = index_flat
                                        .iter_dim(0)
                                        .map(|idxs| {
                                            let idxs = idxs.squeeze::<1>(0);
                                            Tensor::select(#input.clone(), #dim, idxs)
                                        })
                                        .collect();
                                    let #output = Tensor::stack::<#final_rank>(out, #dim);
                                },
                            }
                        }
                        GatherIndices::Static(indices) => {
                            // Static indices for tensor gathering
                            let indices_tokens = indices
                                .iter()
                                .map(|&idx| quote! { #idx })
                                .collect::<Vec<_>>();

                            // Calculate output rank based on indices dimensionality
                            let indices_rank = 1; // Static indices are always 1D
                            let output_rank = indices_rank + input_rank - 1;

                            if output_rank == 0 {
                                quote! {
                                    let indices = Tensor::<B, 1, _>::from_data([#(#indices_tokens),*], &*self.device);
                                    let #output = Tensor::select(#input, #dim, indices);
                                }
                            } else {
                                quote! {
                                    let indices = Tensor::<B, 1, _>::from_data([#(#indices_tokens),*], &*self.device);
                                    let #output = Tensor::select(#input, #dim, indices);
                                }
                            }
                        }
                        _ => panic!("Gather needs Scalar or Tensor index, got {:?}!", self.index),
                    }
                }
                _ => panic!("Gather needs Tensor output, got {:?}!", self.output),
            }
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // No special imports needed for tensor/shape outputs
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, ShapeType, TensorType,
        graph::BurnGraph,
        node::{gather::GatherNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_gather_1d_idx() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_int("tensor2", 1)),
            Type::Tensor(TensorType::new_float("tensor3", 2)),
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
    fn test_codegen_gather_2d_idx() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_int("tensor2", 2)),
            Type::Tensor(TensorType::new_float("tensor3", 3)),
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

                    let n_dims = indices.dims().len();
                    let index_flat = match n_dims {
                        1 => indices.reshape([1, -1]),
                        n if n >= 2 => indices.flatten::<2>(0, n - 2),
                        _ => panic!("Number of dimensions must be greater than 0"),
                    };

                    let out = index_flat
                        .iter_dim(0)
                        .map(|idxs| {
                            let idxs = idxs.squeeze::<1>(0);
                            Tensor::select(tensor1.clone(), 0, idxs)
                        })
                        .collect();
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
            Type::Shape(ShapeType::new("shape2", 1)),
            0,
        ));

        graph.register_input_output(
            vec!["shape1".to_string(), "tensor1".to_string()],
            vec!["shape2".to_string()],
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
                ) -> [usize; 1] {
                    let input_shape = &shape1;
                    let indices_data = tensor1.to_data();
                    let indices_vec: alloc::vec::Vec<usize> = indices_data.iter::<i64>().map(|x| x as usize).collect();
                    let mut output_shape = alloc::vec::Vec::with_capacity(indices_vec.len());
                    for &idx in &indices_vec {
                        output_shape.push(input_shape[idx]);
                    }
                    let shape2: [usize; 1usize] = output_shape.try_into().unwrap();
                    shape2
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
            Type::Tensor(TensorType::new_float("tensor2", 1)),
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

    // Scalar output test removed - no longer supported

    #[test]
    fn test_codegen_gather_static_indices() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::with_static_indices(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            vec![0, 2, 1],
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            0,
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                    tensor1: Tensor<B, 2>
                ) -> Tensor<B, 2> {
                    let indices = Tensor::<B, 1, _>::from_data([0i64, 2i64, 1i64], &*self.device);
                    let tensor2 = Tensor::select(tensor1, 0, indices);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_gather_static_shape_indices() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::with_static_indices(
            Type::Shape(ShapeType::new("shape1", 4)),
            vec![0, 2],
            Type::Shape(ShapeType::new("shape2", 2)),
            0,
        ));

        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

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
                    shape1: [usize; 4]
                ) -> [usize; 2] {
                    let input_shape = &shape1;
                    let shape2: [usize; 2usize] = [input_shape[0usize], input_shape[2usize]];
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_gather_scalar_from_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::with_static_indices(
            Type::Shape(ShapeType::new("shape1", 4)),
            vec![1],
            Type::Scalar(ScalarType::new("dim1", ScalarKind::Int64)),
            0,
        ));

        graph.register_input_output(vec!["shape1".to_string()], vec!["dim1".to_string()]);

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
                    shape1: [usize; 4]
                ) -> i64 {
                    let input_shape = &shape1;
                    let dim1 = input_shape[1usize];
                    dim1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
