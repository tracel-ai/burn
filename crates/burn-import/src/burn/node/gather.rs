use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens, scalar_type_tokens};

use burn::record::PrecisionSettings;
use onnx_ir::{ArgType, Argument};
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::gather::GatherNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> proc_macro2::TokenStream {
        let input_arg = self.inputs.first().unwrap();

        match &input_arg.ty {
            ArgType::Shape(_) => forward_shape_gather(self),
            ArgType::Tensor(_) => forward_tensor_gather(self, scope, node_position),
            _ => panic!(
                "Gather needs Tensor or Shape input, got {:?}!",
                input_arg.ty
            ),
        }
    }
}

fn forward_shape_gather(node: &onnx_ir::gather::GatherNode) -> proc_macro2::TokenStream {
    let input_arg = node.inputs.first().unwrap();
    let index_arg = &node.inputs[1];
    let output_arg = node.outputs.first().unwrap();

    let input_shape_name = arg_to_ident(input_arg);
    let output = arg_to_ident(output_arg);

    match &output_arg.ty {
        ArgType::Scalar(dtype) => {
            // Gathering a single element from a shape produces a scalar
            let scalar_ty = scalar_type_tokens(dtype);
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    let index = arg_to_ident(index_arg);
                    // Handle negative indices properly for runtime scalars
                    quote! {
                        let actual_idx = if #index < 0 {
                            (#input_shape_name.len() as i64 + #index) as usize
                        } else {
                            #index as usize
                        };
                        let #output = #input_shape_name[actual_idx] as #scalar_ty;
                    }
                }
                _ => panic!(
                    "Gather from Shape to Scalar needs scalar index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        ArgType::Shape(out_rank) => {
            match &index_arg.ty {
                ArgType::Tensor(idx_tensor) => {
                    let index = arg_to_ident(index_arg);
                    let index_rank = idx_tensor.rank;
                    let output_rank = out_rank;

                    if index_rank == 1 {
                        // Handle negative indices properly for runtime tensors
                        quote! {
                            let #output: [i64; #output_rank] = #index.to_data()
                                .iter::<i64>()
                                .map(|idx| {
                                    let actual_idx = if idx < 0 {
                                        (#input_shape_name.len() as i64 + idx) as usize
                                    } else {
                                        idx as usize
                                    };
                                    #input_shape_name[actual_idx]
                                })
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
                ArgType::Shape(_idx_rank) => {
                    // Shape indices for gathering from Shape
                    let index_name = arg_to_ident(index_arg);
                    let output_rank = out_rank;

                    // Handle negative indices properly for runtime shape indices
                    quote! {
                        let #output: [i64; #output_rank] = #index_name
                            .iter()
                            .map(|&idx| {
                                let actual_idx = if idx < 0 {
                                    (#input_shape_name.len() as i64 + idx) as usize
                                } else {
                                    idx as usize
                                };
                                #input_shape_name[actual_idx]
                            })
                            .collect::<alloc::vec::Vec<_>>()
                            .try_into()
                            .unwrap();
                    }
                }
                _ => panic!(
                    "Gather from Shape to Shape needs Tensor or Shape index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        _ => panic!(
            "Gather from Shape input can only output Shape or Scalar, got {:?}!",
            output_arg.ty
        ),
    }
}

fn forward_tensor_gather(
    node: &onnx_ir::gather::GatherNode,
    scope: &mut Scope,
    node_position: usize,
) -> proc_macro2::TokenStream {
    let dim = node.config.axis.to_tokens();
    let input_arg = node.inputs.first().unwrap();
    let index_arg = &node.inputs[1];
    let output_arg = node.outputs.first().unwrap();

    let input_rank = match &input_arg.ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => unreachable!(),
    };
    let input = scope.tensor_use_owned(input_arg, node_position);
    let output = arg_to_ident(output_arg);

    match &output_arg.ty {
        ArgType::Scalar(dtype) => {
            // Gathering a single element from a tensor produces a scalar
            let scalar_ty = scalar_type_tokens(dtype);
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    let index = arg_to_ident(index_arg);
                    quote! {
                        let indices = Tensor::<B, 1, _>::from_data([#index], &*self.device);
                        let selected = Tensor::select(#input, #dim, indices);
                        let #output = selected.into_scalar().elem::<#scalar_ty>();
                    }
                }
                _ => panic!(
                    "Gather from Tensor to Scalar needs scalar index, got {:?}!",
                    index_arg.ty
                ),
            }
        }
        ArgType::Tensor(_) => {
            match &index_arg.ty {
                ArgType::Scalar(_) => {
                    // Use tensor.slice(...) with range syntax for more efficient gather operation
                    let index = arg_to_ident(index_arg);
                    let output_rank = input_rank - 1;

                    // Generate slice ranges: s![.., index..index+1, ..] where the range is at position `dim`
                    let slice_args = (0..input_rank)
                        .map(|i| {
                            if i == node.config.axis {
                                quote! { (#index as usize)..((#index as usize) + 1) }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect::<Vec<_>>();

                    quote! {
                        let sliced = #input.slice(s![#(#slice_args),*]);
                        let #output = sliced.squeeze_dim::<#output_rank>(#dim);
                    }
                }
                ArgType::Tensor(idx_tensor) => {
                    let index = scope.tensor_use_owned(index_arg, node_position);
                    let index_rank = idx_tensor.rank;
                    let output_rank = index_rank + input_rank - 1;
                    let final_rank = output_rank.max(1); // Ensure minimum rank of 1

                    // Use proc_macro2::Literal to avoid usize suffix
                    let index_rank_lit = proc_macro2::Literal::usize_unsuffixed(index_rank);
                    let final_rank_lit = proc_macro2::Literal::usize_unsuffixed(final_rank);

                    quote! {
                        let #output = #input.take::<#index_rank_lit, #final_rank_lit>(#dim, #index);
                    }
                }
                ArgType::Shape(_) => {
                    let shape_name = arg_to_ident(index_arg);

                    // Shape array can be directly used to create tensor data
                    quote! {
                        let indices = Tensor::<B, 1, _>::from_data(#shape_name, &*self.device);
                        let #output = Tensor::select(#input, #dim, indices);
                    }
                }
            }
        }
        _ => panic!("Gather needs Tensor output, got {:?}!", output_arg.ty),
    }
}
