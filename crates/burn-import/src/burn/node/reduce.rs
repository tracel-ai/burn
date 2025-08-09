use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::reduce::ReduceConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Min,
    Max,
    Sum,
    Prod,
    Mean,
    L1,
    L2,
    LogSum,
    LogSumExp,
    SumSquare,
}

impl ReductionType {
    /// Generate the code for a reduction operation along all dimensions.
    fn try_forward_reduce(&self, input: TokenStream) -> Option<TokenStream> {
        match self {
            ReductionType::Min => Some(quote! { #input.min() }),
            ReductionType::Max => Some(quote! { #input.max() }),
            ReductionType::Sum => Some(quote! { #input.sum() }),
            ReductionType::Prod => Some(quote! { #input.prod() }),
            ReductionType::Mean => Some(quote! { #input.mean() }),
            _ => None,
        }
    }

    /// Generate the code for a reduction operation along a specific dimension.
    fn forward_reduce_by_dim(&self, input: TokenStream, dim: usize) -> TokenStream {
        match self {
            ReductionType::Min => quote! { #input.min_dim(#dim) },
            ReductionType::Max => quote! { #input.max_dim(#dim) },
            ReductionType::Sum => quote! { #input.sum_dim(#dim) },
            ReductionType::Prod => quote! { #input.prod_dim(#dim) },
            ReductionType::Mean => quote! { #input.mean_dim(#dim) },
            _ => panic!("Unsupported reduction type {:?}", self),
        }
    }
}

#[derive(Debug, Clone, new)]
pub struct ReduceNode {
    pub input: TensorType,
    pub output: TensorType,
    pub reduction_type: ReductionType,
    pub config: ReduceConfig,
}

impl ReduceNode {
    fn reduce_by_dims(
        reduction_type: ReductionType,
        input: TokenStream,
        dims: Vec<usize>,
        keepdims: bool,
    ) -> TokenStream {
        // Reducing along specified dimensions
        let reduced_input = dims.iter().fold(input, |tokens, dim| {
            reduction_type.forward_reduce_by_dim(tokens, *dim)
        });

        if keepdims {
            reduced_input
        } else {
            // Squeezing dimensions
            let dims = dims.to_tokens();
            quote! { #reduced_input.squeeze_dims(&#dims) }
        }
    }

    fn forward_reduce(
        reduction_type: ReductionType,
        input: TokenStream,
        mut dims: Vec<usize>,
        keepdims: bool,
        input_rank: usize,
        output_rank: usize,
    ) -> TokenStream {
        if dims.is_empty() {
            if let Some(reduced_input) = reduction_type.try_forward_reduce(input.clone()) {
                // Reducing along all dimensions
                if keepdims {
                    quote! { #reduced_input.expand([1; #output_rank]) }
                } else {
                    reduced_input
                }
            } else {
                // Reducing along all specific dimensions
                if dims.is_empty() {
                    dims = (0..input_rank).collect();
                }
                Self::reduce_by_dims(reduction_type, input, dims, keepdims)
            }
        } else {
            // Reducing along specific dimensions
            Self::reduce_by_dims(reduction_type, input, dims, keepdims)
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReduceNode {
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let input_rank = self.input.rank;
        let output_rank = self.output.rank;
        let dims = self.config.dims.clone();
        let keepdims = self.config.keepdims;

        let output_expr = match self.reduction_type {
            ReductionType::SumSquare => {
                let input_square = quote! { #input.powi_scalar(2) };
                let input_square_reduced = Self::forward_reduce(
                    ReductionType::Sum,
                    input_square,
                    dims,
                    keepdims,
                    input_rank,
                    output_rank,
                );
                input_square_reduced
            }
            ReductionType::L1 => {
                let input_abs = quote! { #input.abs() };
                let input_abs_reduced = Self::forward_reduce(
                    ReductionType::Sum,
                    input_abs,
                    dims,
                    keepdims,
                    input_rank,
                    output_rank,
                );
                input_abs_reduced
            }
            ReductionType::L2 => {
                let input_square = quote! { #input.powi_scalar(2) };
                let input_square_reduced = Self::forward_reduce(
                    ReductionType::Sum,
                    input_square,
                    dims,
                    keepdims,
                    input_rank,
                    output_rank,
                );

                match self.input.kind {
                    TensorKind::Int => {
                        quote! { #input_square_reduced.float().cast(burn::tensor::DType::F32).sqrt().int() }
                    }
                    TensorKind::Float => {
                        quote! {
                            let input_dtype = #input.dtype();
                            #input_square_reduced.cast(burn::tensor::DType::F32).sqrt().cast(input_dtype)
                        }
                    }
                    _ => panic!("Unsupported input type for L2 reduction"),
                }
            }
            ReductionType::LogSum => {
                let input_reduced = Self::forward_reduce(
                    ReductionType::Sum,
                    input.clone(),
                    dims,
                    keepdims,
                    input_rank,
                    output_rank,
                );

                match self.input.kind {
                    TensorKind::Int => {
                        quote! { #input_reduced.float().cast(burn::tensor::DType::F32).log().int() }
                    }
                    TensorKind::Float => {
                        quote! {
                            let input_dtype = #input.dtype();
                            #input_reduced.cast(burn::tensor::DType::F32).log().cast(input_dtype)
                        }
                    }
                    _ => panic!("Unsupported input type for LogSum reduction"),
                }
            }
            ReductionType::LogSumExp => {
                let input_double = match self.input.kind {
                    TensorKind::Int => {
                        quote! { #input.float().cast(burn::tensor::DType::F32) }
                    }
                    TensorKind::Float => {
                        quote! { #input.cast(burn::tensor::DType::F32) }
                    }
                    _ => panic!("Unsupported input type for LogSumExp reduction"),
                };

                let input_max_reduced = Self::forward_reduce(
                    ReductionType::Max,
                    quote! { input_double.clone() },
                    dims.clone(),
                    keepdims,
                    input_rank,
                    output_rank,
                );

                let exp_reduced = Self::forward_reduce(
                    ReductionType::Sum,
                    quote! { input_exp_reduced },
                    dims,
                    keepdims,
                    input_rank,
                    output_rank,
                );

                let input_reduced = quote! {
                    let input_dtype = #input.dtype();
                    let input_shape = #input.shape();
                    let input_double = #input_double;
                    let input_max_reduced = #input_max_reduced;
                    let input_exp_reduced = (input_double - input_max_reduced.clone().expand(input_shape)).exp();
                    let exp_sum_reduced = #exp_reduced;
                    (input_max_reduced + exp_sum_reduced.log())
                };

                match self.input.kind {
                    TensorKind::Int => {
                        quote! { #input_reduced.int() }
                    }
                    TensorKind::Float => {
                        quote! { #input_reduced.cast(input_dtype) }
                    }
                    _ => panic!("Unsupported input type for LogSumExp reduction"),
                }
            }
            _ => Self::forward_reduce(
                self.reduction_type,
                input,
                dims,
                keepdims,
                input_rank,
                output_rank,
            ),
        };

        quote! { let #output = { #output_expr }; }
    }

    fn into_node(self) -> Node<PS> {
        Node::Reduce(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;
    use onnx_ir::{ArgType, Argument, ElementType};

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{reduce::ReduceNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_reduce_max() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 2),
            ReductionType::Max,
            ReduceConfig::new(vec![0, 2], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 2> {
                    let tensor2 = { 
                        tensor1
                            .max_dim(0usize)
                            .max_dim(2usize)
                            .squeeze_dims(&[0, 2]) 
                    };
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_min() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ReductionType::Min,
            ReduceConfig::new(vec![1, 3], true),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = { tensor1.min_dim(1usize).min_dim(3usize) };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_sum() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 1),
            ReductionType::Sum,
            ReduceConfig::new(vec![], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = { tensor1.sum() };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_sum_square() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ReductionType::SumSquare,
            ReduceConfig::new(vec![], true),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = { tensor1.powi_scalar(2).sum().expand([1; 4usize]) };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_l1() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 3),
            ReductionType::L1,
            ReduceConfig::new(vec![0], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 3> {
                    let tensor2 = { tensor1.abs().sum_dim(0usize).squeeze_dims(&[0]) };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_l2() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 2),
            ReductionType::L2,
            ReduceConfig::new(vec![0, 3], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 2> {
                    let tensor2 = {
                        let input_dtype = tensor1.dtype();
                        tensor1
                            .powi_scalar(2)
                            .sum_dim(0usize)
                            .sum_dim(3usize)
                            .squeeze_dims(&[0, 3])
                            .cast(burn::tensor::DType::F32)
                            .sqrt()
                            .cast(input_dtype)
                    };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_log_sum() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 3),
            ReductionType::LogSum,
            ReduceConfig::new(vec![0], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 3> {
                    let tensor2 = {
                        let input_dtype = tensor1.dtype();
                        tensor1
                            .sum_dim(0usize)
                            .squeeze_dims(&[0])
                            .cast(burn::tensor::DType::F32)
                            .log()
                            .cast(input_dtype)
                    };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_prod() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ReductionType::Prod,
            ReduceConfig::new(vec![3], true),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = { tensor1.prod_dim(3usize) };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_mean() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 1),
            ReductionType::Mean,
            ReduceConfig::new(vec![], false),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = { tensor1.mean() };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reduce_log_sum_exp() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReduceNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ReductionType::LogSumExp,
            ReduceConfig::new(vec![2], true),
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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = {
                        let input_dtype = tensor1.dtype();
                        let input_shape = tensor1.shape();
                        let input_double = tensor1.cast(burn::tensor::DType::F32);
                        let input_max_reduced = input_double.clone().max_dim(2usize);
                        let input_exp_reduced =
                            (input_double - input_max_reduced.clone().expand(input_shape)).exp();
                        let exp_sum_reduced = input_exp_reduced.sum_dim(2usize);
                        (input_max_reduced + exp_sum_reduced.log()).cast(input_dtype)
                    };

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
