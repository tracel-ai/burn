use super::prelude::*;

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

    fn reduce_by_dims(
        &self,
        input: TokenStream,
        dims: Vec<usize>,
        keepdims: bool,
        output_rank: usize,
    ) -> TokenStream {
        // Reducing along specified dimensions
        let reduced_input = dims.iter().fold(input, |tokens, dim| {
            self.forward_reduce_by_dim(tokens, *dim)
        });

        if keepdims {
            reduced_input
        } else {
            // Check if the result should be a scalar (rank 0)
            if output_rank == 0 {
                // For scalar outputs, we don't need to squeeze dims as the reduce operations
                // should already return the correct rank
                reduced_input
            } else {
                // Squeezing dimensions for non-scalar outputs
                let dims = dims.to_tokens();
                quote! { #reduced_input.squeeze_dims(&#dims) }
            }
        }
    }

    fn forward_reduce(
        &self,
        input: TokenStream,
        mut dims: Vec<usize>,
        keepdims: bool,
        input_rank: usize,
        output_rank: usize,
    ) -> TokenStream {
        if dims.is_empty() {
            if let Some(reduced_input) = self.try_forward_reduce(input.clone()) {
                // Reducing along all dimensions
                if keepdims {
                    quote! { #reduced_input.expand([1; #output_rank]) }
                } else {
                    reduced_input
                }
            } else {
                // Reducing along all specific dimensions
                dims = (0..input_rank).collect();
                self.reduce_by_dims(input, dims, keepdims, output_rank)
            }
        } else {
            // Reducing along specific dimensions
            self.reduce_by_dims(input, dims, keepdims, output_rank)
        }
    }
}

// Helper macro to implement NodeCodegen for reduce nodes
macro_rules! impl_reduce_node {
    ($node_type:ty, $reduction_type:expr) => {
        impl<PS: PrecisionSettings> NodeCodegen<PS> for $node_type {
            fn inputs(&self) -> &[Argument] {
                &self.inputs
            }

            fn outputs(&self) -> &[Argument] {
                &self.outputs
            }

            fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
                let input_arg = self.inputs.first().unwrap();
                let output_arg = self.outputs.first().unwrap();

                // Get input rank and check if it's boolean
                let (input_rank, is_bool) = match &input_arg.ty {
                    onnx_ir::ir::ArgType::Tensor(tensor) => {
                        (tensor.rank, tensor.dtype == onnx_ir::ir::DType::Bool)
                    }
                    _ => panic!("Reduce node input must be a tensor"),
                };

                // Get output rank
                let output_rank = match &output_arg.ty {
                    onnx_ir::ir::ArgType::Tensor(tensor) => tensor.rank,
                    onnx_ir::ir::ArgType::Scalar(_) => 0,
                    _ => panic!("Reduce node output must be tensor or scalar"),
                };

                let input = scope.arg(input_arg);
                let output = arg_to_ident(output_arg);

                let dims = self.config.dims.clone();
                let keepdims = self.config.keepdims;

                // For boolean tensors with Min/Max reduction, use all()/any()
                if is_bool && matches!($reduction_type, ReductionType::Min | ReductionType::Max) {
                    let (bool_reduction_all, bool_reduction_dim) = match $reduction_type {
                        ReductionType::Min => (quote! { all }, quote! { all_dim }),
                        ReductionType::Max => (quote! { any }, quote! { any_dim }),
                        _ => unreachable!(),
                    };

                    let reduced_input = if dims.is_empty() {
                        // Reduce all dimensions
                        quote! { #input.#bool_reduction_all() }
                    } else {
                        // Reduce along specific dimensions
                        dims.iter().fold(input.clone(), |tokens, dim| {
                            quote! { #tokens.#bool_reduction_dim(#dim) }
                        })
                    };

                    let final_output = if keepdims {
                        if dims.is_empty() {
                            quote! { #reduced_input.expand([1; #output_rank]) }
                        } else {
                            reduced_input
                        }
                    } else if output_rank == 0 {
                        reduced_input
                    } else {
                        // Need to squeeze dimensions
                        let dims_to_squeeze = dims.to_tokens();
                        quote! { #reduced_input.squeeze_dims(&#dims_to_squeeze) }
                    };

                    return if output_rank == 0 {
                        quote! {
                            let #output = {
                                #final_output.into_scalar().elem::<bool>()
                            };
                        }
                    } else {
                        quote! {
                            let #output = #final_output;
                        }
                    };
                }

                let raw_output_expr = match $reduction_type {
                    ReductionType::SumSquare => {
                        let input_square = quote! { #input.square() };
                        ReductionType::Sum.forward_reduce(
                            input_square,
                            dims.clone(),
                            keepdims,
                            input_rank,
                            output_rank,
                        )
                    }
                    ReductionType::L1 => {
                        let input_abs = quote! { #input.abs() };
                        ReductionType::Sum.forward_reduce(
                            input_abs,
                            dims.clone(),
                            keepdims,
                            input_rank,
                            output_rank,
                        )
                    }
                    ReductionType::L2 => {
                        let input_square = quote! { #input.square() };
                        let input_square_reduced = ReductionType::Sum.forward_reduce(
                            input_square,
                            dims.clone(),
                            keepdims,
                            input_rank,
                            output_rank,
                        );

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64 => {
                                        // Cast to F32 before sqrt to avoid overflow/underflow
                                        quote! { #input_square_reduced.float().cast(burn::tensor::DType::F32).sqrt().int() }
                                    }
                                    _ => {
                                        // Float types - cast to F32 before sqrt, then back
                                        quote! {
                                            let input_dtype = #input.dtype();
                                            #input_square_reduced.cast(burn::tensor::DType::F32).sqrt().cast(input_dtype)
                                        }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    ReductionType::LogSum => {
                        let input_reduced = ReductionType::Sum.forward_reduce(
                            input.clone(),
                            dims.clone(),
                            keepdims,
                            input_rank,
                            output_rank,
                        );

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64 => {
                                        quote! { #input_reduced.float().cast(burn::tensor::DType::F32).log().int() }
                                    }
                                    _ => {
                                        quote! {
                                            let input_dtype = #input.dtype();
                                            #input_reduced.cast(burn::tensor::DType::F32).log().cast(input_dtype)
                                        }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    ReductionType::LogSumExp => {
                        let input_double = match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64 => {
                                        quote! { #input.float().cast(burn::tensor::DType::F32) }
                                    }
                                    _ => {
                                        quote! { #input.cast(burn::tensor::DType::F32) }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        };

                        let input_max_reduced = ReductionType::Max.forward_reduce(
                            quote! { input_double.clone() },
                            dims.clone(),
                            keepdims,
                            input_rank,
                            output_rank,
                        );

                        let exp_reduced = ReductionType::Sum.forward_reduce(
                            quote! { input_exp_reduced },
                            dims.clone(),
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

                        match &input_arg.ty {
                            onnx_ir::ir::ArgType::Tensor(tensor) => {
                                match tensor.dtype {
                                    onnx_ir::ir::DType::I8
                                    | onnx_ir::ir::DType::I16
                                    | onnx_ir::ir::DType::I32
                                    | onnx_ir::ir::DType::I64 => {
                                        quote! { #input_reduced.int() }
                                    }
                                    _ => {
                                        quote! { #input_reduced.cast(input_dtype) }
                                    }
                                }
                            }
                            _ => panic!("Reduce node input must be a tensor"),
                        }
                    }
                    _ => $reduction_type.forward_reduce(
                        input,
                        dims.clone(),
                        keepdims,
                        input_rank,
                        output_rank,
                    ),
                };

                // Handle scalar outputs by extracting the scalar value from the tensor result
                let output_expr = match &output_arg.ty {
                    onnx_ir::ir::ArgType::Scalar(dtype) => {
                        // For scalar outputs, extract the scalar value using .into_scalar()
                        match dtype {
                            onnx_ir::ir::DType::I8 => quote! { #raw_output_expr.into_scalar().elem::<i8>() },
                            onnx_ir::ir::DType::I16 => quote! { #raw_output_expr.into_scalar().elem::<i16>() },
                            onnx_ir::ir::DType::I32 => quote! { #raw_output_expr.into_scalar().elem::<i32>() },
                            onnx_ir::ir::DType::I64 => quote! { #raw_output_expr.into_scalar().elem::<i64>() },
                            onnx_ir::ir::DType::F16 => quote! { #raw_output_expr.into_scalar().elem::<half::f16>() },
                            onnx_ir::ir::DType::F32 => quote! { #raw_output_expr.into_scalar().elem::<f32>() },
                            onnx_ir::ir::DType::F64 => quote! { #raw_output_expr.into_scalar().elem::<f64>() },
                            onnx_ir::ir::DType::Bool => quote! { #raw_output_expr.into_scalar().elem::<bool>() },
                            _ => panic!("Unsupported scalar type for reduce output"),
                        }
                    }
                    onnx_ir::ir::ArgType::Tensor(_) => raw_output_expr,
                    _ => panic!("Reduce node output must be tensor or scalar"),
                };

                quote! { let #output = { #output_expr }; }
            }

            fn register_imports(&self, _imports: &mut BurnImports) {
                // No special imports needed for reduce operations
            }
        }
    };
}

// Implement NodeCodegen for all reduce node types
impl_reduce_node!(onnx_ir::node::reduce::ReduceMaxNode, ReductionType::Max);
impl_reduce_node!(onnx_ir::node::reduce::ReduceMinNode, ReductionType::Min);
impl_reduce_node!(onnx_ir::node::reduce::ReduceSumNode, ReductionType::Sum);
impl_reduce_node!(onnx_ir::node::reduce::ReduceProdNode, ReductionType::Prod);
impl_reduce_node!(onnx_ir::node::reduce::ReduceMeanNode, ReductionType::Mean);
impl_reduce_node!(onnx_ir::node::reduce::ReduceL1Node, ReductionType::L1);
impl_reduce_node!(onnx_ir::node::reduce::ReduceL2Node, ReductionType::L2);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceLogSumNode,
    ReductionType::LogSum
);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceLogSumExpNode,
    ReductionType::LogSumExp
);
impl_reduce_node!(
    onnx_ir::node::reduce::ReduceSumSquareNode,
    ReductionType::SumSquare
);
