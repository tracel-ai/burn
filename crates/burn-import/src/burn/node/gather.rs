use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, ScalarKind, ToTokens, Type};

use burn::record::PrecisionSettings;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct GatherNode {
    pub input: Type,
    pub index: Type,
    pub output: Type,
    pub dim: usize,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for GatherNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
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
            Type::Tensor(in_tensor) => in_tensor.rank,
            Type::Shape(_) => 1,
            _ => panic!("Gather needs Tensor or Shape input, got {:?}!", self.input),
        };

        let input = match &self.input {
            Type::Tensor(in_tensor) => scope.tensor_use_owned(in_tensor, node_position),
            Type::Shape(in_shape) => in_shape.to_tensor(),
            _ => panic!("Gather needs Scalar or Shape input, got {:?}!", self.input),
        };

        let output = &self.output.name();

        match &self.output {
            Type::Scalar(sc) => {
                assert_eq!(input_rank, 1);
                let index = match &self.index {
                    Type::Scalar(idx) => idx.name.clone(),
                    _ => panic!("Gather needs Scalar index, got {:?}!", self.index),
                };
                let scalar_kind = &sc.kind;
                match scalar_kind {
                    ScalarKind::Int32 => quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let gathered = Tensor::select(#input, #dim, indices);
                        let #output = gathered.into_scalar().to_i32();
                        #output
                    },
                    ScalarKind::Int64 => quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let gathered = Tensor::select(#input, #dim, indices);
                        let #output = gathered.into_scalar().to_i64();
                    },
                    ScalarKind::Float32 => quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let gathered = Tensor::select(#input, #dim, indices);
                        let #output = gathered.into_scalar().to_f32();
                    },
                    ScalarKind::Float64 => quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let gathered = Tensor::select(#input, #dim, indices);
                        let #output = gathered.into_scalar().to_f64();
                    },
                    ScalarKind::Bool => quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let gathered = Tensor::select(#input, #dim, indices);
                        let #output = gathered.into_scalar().to_bool();
                    },
                }
            }
            Type::Tensor(_) => {
                match &self.index {
                    Type::Scalar(idx_scalar) => {
                        // To do a scalar select (select just a single index in one dim),
                        // convert the 0-D index to a 1-D Tensor with len 1 to use burn's select,
                        // then squeeze the dimension to reduce the rank
                        let index = &idx_scalar.name;
                        let output_rank = input_rank - 1;
                        quote! {
                            let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                            let slice = Tensor::select(#input, #dim, indices);
                            let #output = slice.squeeze::<#output_rank>(#dim);
                        }
                    }
                    Type::Tensor(idx_tensor) => {
                        let index = scope.tensor_use_owned(idx_tensor, node_position);
                        let index_rank = idx_tensor.rank;
                        let output_rank = index_rank + input_rank - 1;
                        match index_rank {
                            1 => quote! {
                                let indices = #index;
                                let #output = Tensor::select(#input, #dim, indices);
                            },
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
                                let #output = Tensor::stack::<#output_rank>(out, #dim);
                            },
                        }
                    }
                    _ => panic!("Gather needs Scalar or Tensor index, got {:?}!", self.index),
                }
            }
            _ => panic!(
                "Gather needs Scalar or Tensor output, got {:?}!",
                self.output
            ),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Gather(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.output {
            Type::Scalar(_) => {
                imports.register("burn::tensor::cast::ToElement");
            }
            _ => {}
        }
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
            Type::Tensor(TensorType::new_int("tensor2", 1)),
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
                        Tensor::<B, 1, burn::tensor::Int>::from_data(&shape1 as &[_], &*self.device),
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

    #[test]
    fn test_codegen_gather_scalar_output() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(GatherNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 1)),
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int64)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Int64)),
            0,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "scalar1".to_string()],
            vec!["scalar2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::cast::ToElement;
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
                    scalar1: i64
                ) -> i64 {
                    let indices = Tensor::<B, 1, _>::from_data([scalar1], &*self.device);
                    let gathered = Tensor::select(tensor1, 0, indices);
                    let scalar2 = gathered.into_scalar().to_i64();
                    scalar2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
