#![allow(clippy::needless_range_loop)]

use super::{NodeCodegen, arg_to_ident};
use crate::burn::{Scope, ToTokens};
use burn::record::PrecisionSettings;
use onnx_ir::{ArgType, Argument};
use proc_macro2::{Literal, TokenStream};
use quote::quote;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::slice::SliceNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let input_arg = self.inputs.first().unwrap();

        match &input_arg.ty {
            ArgType::Tensor(tensor) => {
                generate_tensor_slice(self, input_arg, tensor.rank, scope, node_position, &output)
            }
            ArgType::Shape(shape_rank) => {
                generate_shape_slice(self, input_arg, *shape_rank, &output)
            }
            _ => panic!("Unsupported input type for SliceNode"),
        }
    }
}

fn generate_tensor_slice(
    node: &onnx_ir::slice::SliceNode,
    input_arg: &Argument,
    rank: usize,
    scope: &mut Scope,
    node_position: usize,
    output: &proc_macro2::Ident,
) -> TokenStream {
    let input = scope.tensor_use_owned(input_arg, node_position);
    let mut ranges = vec![quote! { .. }; rank];

    // Build slice ranges based on parameter types
    match (&node.config.starts, &node.config.ends) {
        // Both static: simple case
        (onnx_ir::slice::SliceInput::Static(starts), onnx_ir::slice::SliceInput::Static(ends)) => {
            // Get steps (provided by onnx-ir with ONNX spec defaults)
            let steps = match &node.config.steps {
                Some(onnx_ir::slice::SliceInput::Static(s)) => s,
                _ => panic!("Steps must be Static for static slice"),
            };

            // Check if axes are provided
            if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                // Apply slicing to specified axes (already normalized by onnx-ir)
                for (idx, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
                    if let Some(&axis) = axes.get(idx) {
                        let axis_idx = axis as usize;
                        if axis_idx < rank {
                            let start = start.to_tokens();
                            let step = *steps.get(idx).expect("Step value missing for axis");

                            // Check for i64::MAX which means "to the end"
                            // Slice indices are i32
                            if *end == i64::MAX {
                                if step == 1 {
                                    ranges[axis_idx] = quote! { #start.. };
                                } else {
                                    let step = step.to_tokens();
                                    ranges[axis_idx] = quote! { #start..;#step };
                                }
                            } else if *end > i32::MAX as i64 {
                                panic!("Slice end index {} exceeds i32::MAX", end);
                            } else {
                                let end = end.to_tokens();
                                if step == 1 {
                                    ranges[axis_idx] = quote! { #start..#end };
                                } else {
                                    let step = step.to_tokens();
                                    ranges[axis_idx] = quote! { #start..#end;#step };
                                }
                            }
                        }
                    }
                }
            } else {
                // No axes provided - use default behavior (slice first dimensions)
                let limit = starts.len().min(ends.len()).min(rank);
                for (i, range) in ranges.iter_mut().enumerate().take(limit) {
                    let start = starts[i].to_tokens();
                    let step = *steps.get(i).expect("Step value missing for dimension");

                    // Check for i64::MAX which means "to the end"
                    // Slice indices are i32
                    if ends[i] == i64::MAX {
                        if step == 1 {
                            *range = quote! { #start.. };
                        } else {
                            let step = step.to_tokens();
                            *range = quote! { #start..;#step };
                        }
                    } else if ends[i] > i32::MAX as i64 {
                        panic!("Slice end index {} exceeds i32::MAX", ends[i]);
                    } else {
                        let end = ends[i].to_tokens();
                        if step == 1 {
                            *range = quote! { #start..#end };
                        } else {
                            let step = step.to_tokens();
                            *range = quote! { #start..#end;#step };
                        }
                    }
                }
            }
        }

        // Both runtime shapes: multi-dimensional slicing
        (
            onnx_ir::slice::SliceInput::Runtime(start_ref),
            onnx_ir::slice::SliceInput::Runtime(end_ref),
        ) => {
            let start_arg = &node.inputs[start_ref.input_index];
            let end_arg = &node.inputs[end_ref.input_index];

            if let (ArgType::Shape(start_rank), ArgType::Shape(end_rank)) =
                (&start_arg.ty, &end_arg.ty)
            {
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Check if axes are provided
                if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    let num_dims = axes.len().min(*start_rank).min(*end_rank);
                    for i in 0..num_dims {
                        let axis_idx = axes[i] as usize;
                        if axis_idx < rank {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            ranges[axis_idx] = quote! { #start_name[#idx]..#end_name[#idx] };
                        }
                    }
                } else {
                    // No axes provided - use default behavior
                    let num_dims = start_rank.min(end_rank).min(&rank);
                    for (i, range) in ranges.iter_mut().enumerate().take(*num_dims) {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        *range = quote! { #start_name[#idx]..#end_name[#idx] };
                    }
                }
            } else if matches!(
                (&start_arg.ty, &end_arg.ty),
                (ArgType::Tensor(_), ArgType::Tensor(_))
            ) {
                // Both 1D tensors: extract values from tensors at runtime
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Generate code to extract values from tensors
                let start_data_var = quote! { start_data };
                let start_vec_var = quote! { start_vec };
                let end_data_var = quote! { end_data };
                let end_vec_var = quote! { end_vec };

                // Check if axes are provided (from onnx-ir with ONNX spec defaults)
                if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                    // Build ranges respecting the axes
                    let mut ranges = vec![quote! { .. }; rank];
                    for (idx, &axis) in axes.iter().enumerate() {
                        let axis_idx = axis as usize;
                        if axis_idx < rank {
                            let vec_idx = proc_macro2::Literal::usize_unsuffixed(idx);
                            ranges[axis_idx] = quote! {
                                #start_vec_var[#vec_idx] as usize..#end_vec_var[#vec_idx] as usize
                            };
                        }
                    }

                    return quote! {
                        let #start_data_var = #start_name.to_data();
                        let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                        let #end_data_var = #end_name.to_data();
                        let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                        let #output = #input.slice(s![#(#ranges),*]);
                    };
                } else {
                    // No axes - slice all dimensions (ONNX spec default would have provided axes)
                    panic!("Axes must be provided by onnx-ir for tensor slice");
                }
            } else if matches!(
                (&start_arg.ty, &end_arg.ty),
                (ArgType::Scalar(_), ArgType::Scalar(_))
            ) {
                // Both scalars: use as single axis slice
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Get the axis to slice (provided by onnx-ir with ONNX spec defaults)
                let axis_idx = match &node.config.axes {
                    Some(onnx_ir::slice::SliceInput::Static(axes)) => {
                        *axes.first().expect("Axes array is empty for scalar slice") as usize
                    }
                    _ => panic!("Axes must be Static for scalar slice"),
                };

                if axis_idx < rank {
                    ranges[axis_idx] = quote! { (#start_name as usize)..(#end_name as usize) };
                }
            } else {
                // Mixed types: Shape/Tensor combination
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Determine the number of dims we can safely index
                let start_rank = start_arg.ty.rank();
                let end_rank = end_arg.ty.rank();
                let num_slice_dims = start_rank.min(end_rank);

                // Need to extract tensor values at runtime
                let start_is_tensor = start_arg.ty.is_tensor();
                let end_is_tensor = end_arg.ty.is_tensor();

                let start_vec_var = quote! { start_vec };
                let end_vec_var = quote! { end_vec };

                // Build the extraction code
                let mut extraction_code = quote! {};

                if start_is_tensor {
                    extraction_code = quote! {
                        #extraction_code
                        let start_data = #start_name.to_data();
                        let #start_vec_var: alloc::vec::Vec<i64> = start_data.iter::<i64>().collect();
                    };
                }

                if end_is_tensor {
                    extraction_code = quote! {
                        #extraction_code
                        let end_data = #end_name.to_data();
                        let #end_vec_var: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    };
                }

                // Check if axes are provided
                if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                    // Apply slicing to specified axes only
                    let mut ranges = vec![quote! { .. }; rank];

                    for (i, &axis) in axes.iter().enumerate() {
                        let axis_idx = axis as usize;
                        if axis_idx < rank {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            let start_expr = if start_is_tensor {
                                quote! { #start_vec_var[#idx] as usize }
                            } else {
                                quote! { #start_name[#idx] as usize }
                            };
                            let end_expr = if end_is_tensor {
                                quote! { #end_vec_var[#idx] as usize }
                            } else {
                                quote! { #end_name[#idx] as usize }
                            };
                            ranges[axis_idx] = quote! { #start_expr..#end_expr };
                        }
                    }

                    return quote! {
                        #extraction_code
                        let #output = #input.slice(s![#(#ranges),*]);
                    };
                } else {
                    // No axes provided - apply to first N dimensions
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            if i < num_slice_dims {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                let start_expr = if start_is_tensor {
                                    quote! { #start_vec_var[#idx] as usize }
                                } else {
                                    quote! { #start_name[#idx] as usize }
                                };
                                let end_expr = if end_is_tensor {
                                    quote! { #end_vec_var[#idx] as usize }
                                } else {
                                    quote! { #end_name[#idx] as usize }
                                };
                                quote! { #start_expr..#end_expr }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        #extraction_code
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
            }
        }

        // Static start, runtime end
        (
            onnx_ir::slice::SliceInput::Static(starts),
            onnx_ir::slice::SliceInput::Runtime(end_ref),
        ) => {
            let end_arg = &node.inputs[end_ref.input_index];

            match &end_arg.ty {
                ArgType::Shape(end_rank) => {
                    let end_name = arg_to_ident(end_arg);

                    // Check if axes are provided
                    if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                        // Apply slicing to specified axes (already normalized by onnx-ir)
                        let num_dims = axes.len().min(starts.len()).min(*end_rank);
                        for i in 0..num_dims {
                            let axis_idx = axes[i] as usize;
                            if axis_idx < rank {
                                let start = starts[i].to_tokens();
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                ranges[axis_idx] = quote! { #start..#end_name[#idx] };
                            }
                        }
                    } else {
                        // No axes provided - use default behavior
                        let num_dims = starts.len().min(*end_rank).min(rank);
                        for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                            let start = starts[i].to_tokens();
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            *range = quote! { #start..#end_name[#idx] };
                        }
                    }
                }
                ArgType::Tensor(_) => {
                    // Static start, 1D tensor end
                    let end_name = arg_to_ident(end_arg);

                    // Generate code to extract values from end tensor
                    let end_data_var = quote! { end_data };
                    let end_vec_var = quote! { end_vec };

                    // Build ranges for each dimension
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            if i < starts.len() {
                                let start = Literal::i64_suffixed(starts[i]);
                                quote! {
                                    #start as usize..#end_vec_var[#idx] as usize
                                }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        let #end_data_var = #end_name.to_data();
                        let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
                ArgType::Scalar(_) => {
                    // Static start, scalar end
                    let end_name = arg_to_ident(end_arg);

                    // Get the axis to slice (provided by onnx-ir with ONNX spec defaults)
                    let axis_idx = match &node.config.axes {
                        Some(onnx_ir::slice::SliceInput::Static(axes)) => {
                            *axes.first().expect("Axes array is empty for scalar slice") as usize
                        }
                        _ => panic!("Axes must be Static for scalar slice"),
                    };

                    if axis_idx < rank {
                        // Use the first start value (starts[0]) for the specified axis
                        let start = starts.first().expect("Starts array is empty").to_tokens();
                        ranges[axis_idx] = quote! { #start..(#end_name as usize) };
                    }
                }
            }
        }

        // Runtime start, static end
        (
            onnx_ir::slice::SliceInput::Runtime(start_ref),
            onnx_ir::slice::SliceInput::Static(ends),
        ) => {
            let start_arg = &node.inputs[start_ref.input_index];

            match &start_arg.ty {
                ArgType::Shape(start_rank) => {
                    let start_name = arg_to_ident(start_arg);

                    // Check if axes are provided
                    if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                        // Apply slicing to specified axes (already normalized by onnx-ir)
                        let num_dims = axes.len().min(*start_rank).min(ends.len());
                        for i in 0..num_dims {
                            let axis_idx = axes[i] as usize;
                            if axis_idx < rank {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                let end = ends[i].to_tokens();
                                ranges[axis_idx] = quote! { #start_name[#idx]..#end };
                            }
                        }
                    } else {
                        // No axes provided - use default behavior
                        let ends_len = ends.len();
                        let num_dims = start_rank.min(&ends_len).min(&rank);
                        for (i, range) in ranges.iter_mut().enumerate().take(*num_dims) {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            let end = ends[i].to_tokens();
                            *range = quote! { #start_name[#idx]..#end };
                        }
                    }
                }
                ArgType::Tensor(_) => {
                    // 1D tensor start, static end
                    let start_name = arg_to_ident(start_arg);

                    // Generate code to extract values from start tensor
                    let start_data_var = quote! { start_data };
                    let start_vec_var = quote! { start_vec };

                    // Build ranges for each dimension
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            if i < ends.len() {
                                let end = Literal::i64_suffixed(ends[i]);
                                quote! {
                                    #start_vec_var[#idx] as usize..#end as usize
                                }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        let #start_data_var = #start_name.to_data();
                        let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
                _ => panic!("Unsupported runtime start type for slice"),
            }
        }
    }

    quote! {
        let #output = #input.slice(s![#(#ranges),*]);
    }
}

fn generate_shape_slice(
    node: &onnx_ir::slice::SliceNode,
    input_arg: &Argument,
    shape_rank: usize,
    output: &proc_macro2::Ident,
) -> TokenStream {
    let shape_name = arg_to_ident(input_arg);

    // Get the output rank from the output type
    let output_rank = match &node.outputs.first().unwrap().ty {
        ArgType::Shape(rank) => rank,
        _ => panic!("Expected Shape output type for shape slice operation"),
    };
    let output_rank_lit = Literal::usize_unsuffixed(*output_rank);

    match (&node.config.starts, &node.config.ends) {
        (onnx_ir::slice::SliceInput::Static(starts), onnx_ir::slice::SliceInput::Static(ends))
            if starts.len() == 1 =>
        {
            let start_val = starts[0];
            let end_val = ends[0];

            // Get step value (provided by onnx-ir with ONNX spec defaults)
            let step_val = match &node.config.steps {
                Some(onnx_ir::slice::SliceInput::Static(steps)) => {
                    *steps.first().expect("Steps array is empty")
                }
                _ => panic!("Steps must be Static for shape slice"),
            };

            // Always clamp start/end values
            let shape_len = shape_rank as i64;

            // Handle negative indices and clamp
            let actual_start = if start_val < 0 {
                (shape_len + start_val).max(0) as usize
            } else {
                start_val.min(shape_len) as usize
            };

            let actual_end = if end_val == i64::MAX {
                shape_rank
            } else if end_val < 0 {
                (shape_len + end_val).max(0) as usize
            } else {
                end_val.min(shape_len) as usize
            };

            let start_lit = Literal::usize_unsuffixed(actual_start);
            let end_lit = Literal::usize_unsuffixed(actual_end);

            if step_val == 1 {
                quote! {
                    let #output: [i64; #output_rank_lit] = #shape_name[#start_lit..#end_lit].try_into().unwrap();
                }
            } else if step_val == -1 {
                // For negative step, we need to reverse the slice
                quote! {
                    let #output: [i64; #output_rank_lit] = {
                        let mut slice = #shape_name[#start_lit..#end_lit].to_vec();
                        slice.reverse();
                        slice.try_into().unwrap()
                    };
                }
            } else {
                // For other step values, we need to collect with step
                let step_abs = step_val.abs();
                if step_val > 0 {
                    quote! {
                        let #output: [i64; #output_rank_lit] = #shape_name[#start_lit..#end_lit]
                            .iter()
                            .step_by(#step_abs as usize)
                            .copied()
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap();
                    }
                } else {
                    quote! {
                        let #output: [i64; #output_rank_lit] = #shape_name[#start_lit..#end_lit]
                            .iter()
                            .rev()
                            .step_by(#step_abs as usize)
                            .copied()
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap();
                    }
                }
            }
        }
        _ => {
            // Runtime slicing with scalars
            let (start_expr, end_expr) = get_slice_range_expressions(node);
            let shape_len_lit = Literal::i64_suffixed(shape_rank as i64);

            quote! {
                let start_val = #start_expr as i64;
                let end_val = #end_expr as i64;
                let start_idx = if start_val < 0 { (#shape_len_lit + start_val) as usize } else { start_val as usize };
                let end_idx = if end_val < 0 { (#shape_len_lit + end_val) as usize } else { end_val as usize };
                let #output: [i64; #output_rank_lit] = #shape_name[start_idx..end_idx].try_into().unwrap();
            }
        }
    }
}

fn get_slice_range_expressions(node: &onnx_ir::slice::SliceNode) -> (TokenStream, TokenStream) {
    let start_expr = match &node.config.starts {
        onnx_ir::slice::SliceInput::Static(starts) => starts[0].to_tokens(),
        onnx_ir::slice::SliceInput::Runtime(start_ref) => {
            let start_arg = &node.inputs[start_ref.input_index];
            get_scalar_expr(start_arg)
        }
    };

    let end_expr = match &node.config.ends {
        onnx_ir::slice::SliceInput::Static(ends) => ends[0].to_tokens(),
        onnx_ir::slice::SliceInput::Runtime(end_ref) => {
            let end_arg = &node.inputs[end_ref.input_index];
            get_scalar_expr(end_arg)
        }
    };

    (start_expr, end_expr)
}

fn get_scalar_expr(arg: &Argument) -> TokenStream {
    match &arg.ty {
        ArgType::Scalar(_) => {
            let name = arg_to_ident(arg);
            quote! { #name }
        }
        ArgType::Shape(_) => {
            let name = arg_to_ident(arg);
            // For single-dimension slicing, use the first element of the shape
            quote! { #name[0] }
        }
        ArgType::Tensor(_) => {
            // For 1D tensor, we'll handle it specially in the calling code
            // This shouldn't be called for 1D tensors as they use different logic
            panic!(
                "1D tensor slice parameters should be handled separately, not through get_scalar_expr"
            )
        }
    }
}
