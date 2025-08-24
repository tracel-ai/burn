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

    fn forward_shape_gather(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let input_shape = match &self.input {
            Type::Shape(in_shape) => in_shape,
            _ => unreachable!(),
        };

        match &self.output {
            Type::Scalar(scalar_type) => {
                // Gathering a single element from a shape produces a scalar
                let scalar_ty = scalar_type.ty();
                match &self.index {
                    GatherIndices::Runtime(Type::Scalar(idx_scalar)) => {
                        let index = &idx_scalar.name;
                        let input_shape_name = &input_shape.name;
                        let output = &self.output.name();
                        quote! {
                            let #output = #input_shape_name[#index as usize] as #scalar_ty;
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
                            let #output = #input_shape_name[#idx] as #scalar_ty;
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
                        let output = &self.output.name();

                        if index_rank == 1 {
                            quote! {
                                let #output: [i64; #output_rank] = #index.to_data()
                                    .iter::<i64>()
                                    .map(|idx| #input_shape_name[idx as usize])
                                    .collect::<alloc::vec::Vec<_>>()
                                    .try_into()
                                    .unwrap();
                            }
                        } else {
                            panic!(
                                "Multi-dimensional indices for Shape gather should be 1-dimensional, but got rank {}",
                                index_rank
                            );
                        }
                    }
                    GatherIndices::Static(indices) => {
                        let output_rank = out_shape.rank;
                        let input_shape_name = &input_shape.name;
                        let output = &self.output.name();
                        let indices_len = indices.len();

                        if indices_len != output_rank {
                            panic!(
                                "Static indices length {indices_len} doesn't match output rank {output_rank}"
                            );
                        }

                        // Generate static gathering code
                        let gather_elements = indices.iter().map(|&idx| {
                            let idx_usize = idx as usize;
                            quote! { #input_shape_name[#idx_usize] }
                        });

                        quote! {
                            let #output: [i64; #output_rank] = [#(#gather_elements),*];
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
    }

    fn forward_tensor_gather(
        &self,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> proc_macro2::TokenStream {
        let dim = self.dim.to_tokens();
        let input = match &self.input {
            Type::Tensor(in_tensor) => in_tensor,
            _ => unreachable!(),
        };
        let input_rank = input.rank;
        let input = scope.tensor_use_owned(input, node_position);
        let output = &self.output.name();

        match &self.output {
            Type::Scalar(scalar_type) => {
                // Gathering a single element from a tensor produces a scalar
                let scalar_ty = scalar_type.ty();
                match &self.index {
                    GatherIndices::Runtime(Type::Scalar(idx_scalar)) => {
                        let index = &idx_scalar.name;
                        let output = &scalar_type.name;
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                            let selected = Tensor::select(#input, #dim, indices);
                            let #output = selected.into_scalar().elem::<#scalar_ty>();
                        }
                    }
                    GatherIndices::Static(indices) => {
                        if indices.len() != 1 {
                            panic!(
                                "Static indices length {} doesn't match scalar output",
                                indices.len()
                            );
                        }
                        let idx = indices[0];
                        let output = &scalar_type.name;
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data([#idx], &*self.device);
                            let selected = Tensor::select(#input, #dim, indices);
                            let #output = selected.into_scalar().elem::<#scalar_ty>();
                        }
                    }
                    _ => panic!(
                        "Gather from Tensor to Scalar needs scalar index, got {:?}!",
                        self.index
                    ),
                }
            }
            Type::Tensor(_) => {
                match &self.index {
                    GatherIndices::Runtime(Type::Scalar(idx_scalar)) => {
                        // Use tensor.slice(...) with range syntax for more efficient gather operation
                        let index = &idx_scalar.name;
                        let output_rank = input_rank - 1;

                        // Generate slice ranges: s![.., index..index+1, ..] where the range is at position `dim`
                        let slice_args = (0..input_rank)
                            .map(|i| {
                                if i == self.dim {
                                    quote! { (#index as usize)..((#index as usize) + 1) }
                                } else {
                                    quote! { .. }
                                }
                            })
                            .collect::<Vec<_>>();

                        quote! {
                            let sliced = #input.slice(s![#(#slice_args),*]);
                            let #output = sliced.squeeze::<#output_rank>(#dim);
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
                    GatherIndices::Runtime(Type::Shape(shape_type)) => {
                        let shape_name = &shape_type.name;

                        // Shape array can be directly used to create tensor data
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data(#shape_name, &*self.device);
                            let #output = Tensor::select(#input, #dim, indices);
                        }
                    }
                    GatherIndices::Static(indices) => {
                        // Static indices for tensor gathering
                        let indices_tokens = indices
                            .iter()
                            .map(|&idx| quote! { #idx })
                            .collect::<Vec<_>>();

                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data([#(#indices_tokens),*], &*self.device);
                            let #output = Tensor::select(#input, #dim, indices);
                        }
                    }
                    _ => panic!(
                        "Gather needs Scalar, Tensor, or Shape index, got {:?}!",
                        self.index
                    ),
                }
            }
            _ => panic!("Gather needs Tensor output, got {:?}!", self.output),
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
        match &self.input {
            Type::Shape(_) => self.forward_shape_gather(scope, node_position),
            Type::Tensor(_) => self.forward_tensor_gather(scope, node_position),
            _ => panic!("Gather needs Tensor or Shape input, got {:?}!", self.input),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // s is already available in burn::prelude::*
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
                    shape1: [i64; 3],
                    tensor1: Tensor<B, 1, Int>
                ) -> [i64; 1] {
                    let shape2: [i64; 1usize] = tensor1.to_data()
                        .iter::<i64>()
                        .map(|idx| shape1[idx as usize])
                        .collect::<alloc::vec::Vec<_>>()
                        .try_into()
                        .unwrap();
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
                    tensor1: Tensor<B, 2>,
                    scalar1: i64
                ) -> Tensor<B, 1> {
                    let sliced = tensor1.slice(s![(scalar1 as usize)..((scalar1 as usize) + 1), ..]);
                    let tensor2 = sliced.squeeze::<1usize>(0);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

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
                    shape1: [i64; 4]
                ) -> [i64; 2] {
                    let shape2: [i64; 2usize] = [shape1[0usize], shape1[2usize]];
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
                    shape1: [i64; 4]
                ) -> i64 {
                    let dim1 = shape1[1usize] as i64;
                    dim1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
