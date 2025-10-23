#![allow(clippy::needless_range_loop)]

use super::{NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

#[derive(Debug, Clone)]
pub struct SliceNode {
    pub input: Type,
    pub output: Type,
    pub starts: SliceParam,
    pub ends: SliceParam,
    pub axes: Option<SliceParam>,
    pub steps: Option<SliceParam>,
}

#[derive(Debug, Clone)]
pub enum SliceParam {
    Static(Vec<i64>),
    Runtime(Type),
}

impl SliceNode {
    pub fn new(input: Type, output: Type, starts: SliceParam, ends: SliceParam) -> Self {
        Self {
            input,
            output,
            starts,
            ends,
            axes: None,
            steps: None,
        }
    }

    pub fn with_axes(mut self, axes: SliceParam) -> Self {
        self.axes = Some(axes);
        self
    }

    pub fn with_steps(mut self, steps: SliceParam) -> Self {
        self.steps = Some(steps);
        self
    }

    fn generate_tensor_slice(
        &self,
        tensor: &crate::burn::TensorType,
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let input = scope.tensor_use_owned(tensor, node_position);
        let rank = tensor.rank;
        let mut ranges = vec![quote! { .. }; rank];

        // Build slice ranges based on parameter types
        match (&self.starts, &self.ends) {
            // Both static: simple case
            (SliceParam::Static(starts), SliceParam::Static(ends)) => {
                // Get steps if provided
                let steps = if let Some(SliceParam::Static(ref s)) = self.steps {
                    Some(s)
                } else {
                    None
                };

                // Check if axes are provided
                if let Some(SliceParam::Static(ref axes)) = self.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    for (idx, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
                        if let Some(&axis) = axes.get(idx) {
                            let axis_idx = axis as usize;
                            if axis_idx < rank {
                                let start = start.to_tokens();
                                let step = steps.and_then(|s| s.get(idx)).copied().unwrap_or(1);

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
                        let step = steps.and_then(|s| s.get(i)).copied().unwrap_or(1);

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
                SliceParam::Runtime(Type::Shape(start_shape)),
                SliceParam::Runtime(Type::Shape(end_shape)),
            ) => {
                let start_name = &start_shape.name;
                let end_name = &end_shape.name;

                // Check if axes are provided
                if let Some(SliceParam::Static(ref axes)) = self.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    let num_dims = axes.len().min(start_shape.rank).min(end_shape.rank);
                    for i in 0..num_dims {
                        let axis_idx = axes[i] as usize;
                        if axis_idx < rank {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            ranges[axis_idx] = quote! { #start_name[#idx]..#end_name[#idx] };
                        }
                    }
                } else {
                    // No axes provided - use default behavior
                    let num_dims = start_shape.rank.min(end_shape.rank).min(rank);
                    for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        *range = quote! { #start_name[#idx]..#end_name[#idx] };
                    }
                }
            }

            // Static start, runtime shape end
            (SliceParam::Static(starts), SliceParam::Runtime(Type::Shape(end_shape))) => {
                let end_name = &end_shape.name;

                // Check if axes are provided
                if let Some(SliceParam::Static(ref axes)) = self.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    let num_dims = axes.len().min(starts.len()).min(end_shape.rank);
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
                    let num_dims = starts.len().min(end_shape.rank).min(rank);
                    for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                        let start = starts[i].to_tokens();
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        *range = quote! { #start..#end_name[#idx] };
                    }
                }
            }

            // Runtime shape start, static end
            (SliceParam::Runtime(Type::Shape(start_shape)), SliceParam::Static(ends)) => {
                let start_name = &start_shape.name;

                // Check if axes are provided
                if let Some(SliceParam::Static(ref axes)) = self.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    let num_dims = axes.len().min(start_shape.rank).min(ends.len());
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
                    let num_dims = start_shape.rank.min(ends.len()).min(rank);
                    for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        let end = ends[i].to_tokens();
                        *range = quote! { #start_name[#idx]..#end };
                    }
                }
            }

            // Both 1D tensors: extract values from tensors at runtime
            (
                SliceParam::Runtime(Type::Tensor(start_t)),
                SliceParam::Runtime(Type::Tensor(end_t)),
            ) if start_t.rank == 1 && end_t.rank == 1 => {
                let start_name = &start_t.name;
                let end_name = &end_t.name;

                // Generate code to extract values from tensors
                let input_dims_var = quote! { input_dims };
                let start_data_var = quote! { start_data };
                let start_vec_var = quote! { start_vec };
                let end_data_var = quote! { end_data };
                let end_vec_var = quote! { end_vec };

                // Build ranges for each dimension
                let range_exprs: Vec<_> = (0..rank).map(|i| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    quote! {
                        #start_vec_var.get(#idx).map(|&s| s as usize).unwrap_or(0)..#end_vec_var.get(#idx).map(|&e| e as usize).unwrap_or(#input_dims_var[#idx])
                    }
                }).collect();

                return quote! {
                    let #input_dims_var = #input.dims();
                    let #start_data_var = #start_name.to_data();
                    let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                    let #end_data_var = #end_name.to_data();
                    let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                    let #output = #input.slice(s![#(#range_exprs),*]);
                };
            }

            // Static start, 1D tensor end
            (SliceParam::Static(starts), SliceParam::Runtime(Type::Tensor(end_t)))
                if end_t.rank == 1 =>
            {
                let end_name = &end_t.name;

                // Generate code to extract values from end tensor
                let input_dims_var = quote! { input_dims };
                let end_data_var = quote! { end_data };
                let end_vec_var = quote! { end_vec };

                // Build ranges for each dimension
                let range_exprs: Vec<_> = (0..rank).map(|i| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    if i < starts.len() {
                        let start = Literal::i64_suffixed(starts[i]);
                        quote! {
                            #start as usize..#end_vec_var.get(#idx).map(|&e| e as usize).unwrap_or(#input_dims_var[#idx])
                        }
                    } else {
                        quote! { .. }
                    }
                }).collect();

                return quote! {
                    let #input_dims_var = #input.dims();
                    let #end_data_var = #end_name.to_data();
                    let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                    let #output = #input.slice(s![#(#range_exprs),*]);
                };
            }

            // 1D tensor start, static end
            (SliceParam::Runtime(Type::Tensor(start_t)), SliceParam::Static(ends))
                if start_t.rank == 1 =>
            {
                let start_name = &start_t.name;

                // Generate code to extract values from start tensor
                let input_dims_var = quote! { input_dims };
                let start_data_var = quote! { start_data };
                let start_vec_var = quote! { start_vec };

                // Build ranges for each dimension
                let range_exprs: Vec<_> = (0..rank).map(|i| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    if i < ends.len() {
                        let end = Literal::i64_suffixed(ends[i]);
                        quote! {
                            #start_vec_var.get(#idx).map(|&s| s as usize).unwrap_or(0)..#end as usize
                        }
                    } else {
                        quote! { .. }
                    }
                }).collect();

                return quote! {
                    let #input_dims_var = #input.dims();
                    let #start_data_var = #start_name.to_data();
                    let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                    let #output = #input.slice(s![#(#range_exprs),*]);
                };
            }

            // Shape start, 1D tensor end
            (
                SliceParam::Runtime(Type::Shape(start_shape)),
                SliceParam::Runtime(Type::Tensor(end_t)),
            ) if end_t.rank == 1 => {
                let start_name = &start_shape.name;
                let end_name = &end_t.name;

                // Generate code to extract values from end tensor
                let input_dims_var = quote! { input_dims };
                let end_data_var = quote! { end_data };
                let end_vec_var = quote! { end_vec };

                // Build ranges for each dimension
                let range_exprs: Vec<_> = (0..rank).map(|i| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    if i < start_shape.rank {
                        quote! {
                            #start_name[#idx] as usize..#end_vec_var.get(#idx).map(|&e| e as usize).unwrap_or(#input_dims_var[#idx])
                        }
                    } else {
                        quote! { .. }
                    }
                }).collect();

                return quote! {
                    let #input_dims_var = #input.dims();
                    let #end_data_var = #end_name.to_data();
                    let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                    let #output = #input.slice(s![#(#range_exprs),*]);
                };
            }

            // 1D tensor start, shape end
            (
                SliceParam::Runtime(Type::Tensor(start_t)),
                SliceParam::Runtime(Type::Shape(end_shape)),
            ) if start_t.rank == 1 => {
                let start_name = &start_t.name;
                let end_name = &end_shape.name;

                // Generate code to extract values from start tensor
                let start_data_var = quote! { start_data };
                let start_vec_var = quote! { start_vec };

                // Build ranges for each dimension
                let range_exprs: Vec<_> = (0..rank).map(|i| {
                    let idx = proc_macro2::Literal::usize_unsuffixed(i);
                    if i < end_shape.rank {
                        quote! {
                            #start_vec_var.get(#idx).map(|&s| s as usize).unwrap_or(0)..#end_name[#idx] as usize
                        }
                    } else {
                        quote! { .. }
                    }
                }).collect();

                return quote! {
                    let #start_data_var = #start_name.to_data();
                    let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                    let #output = #input.slice(s![#(#range_exprs),*]);
                };
            }

            // Default: scalar slicing
            _ => {
                let (start_expr, end_expr) = self.get_slice_range_expressions();

                // Check if axes are provided for scalar slicing
                if let Some(SliceParam::Static(ref axes)) = self.axes {
                    if !axes.is_empty() {
                        // Axes are already normalized by onnx-ir
                        let axis_idx = axes[0] as usize;
                        if axis_idx < rank {
                            ranges[axis_idx] = quote! { #start_expr..#end_expr };
                        }
                    } else {
                        // Empty axes array - use first dimension
                        ranges[0] = quote! { #start_expr..#end_expr };
                    }
                } else {
                    // No axes provided - default to first dimension
                    ranges[0] = quote! { #start_expr..#end_expr };
                }
            }
        }

        quote! {
            let #output = #input.slice(s![#(#ranges),*]);
        }
    }

    fn get_slice_range_expressions(&self) -> (TokenStream, TokenStream) {
        let start_expr = match &self.starts {
            SliceParam::Static(starts) => starts[0].to_tokens(),
            SliceParam::Runtime(start_type) => self.get_scalar_expr(start_type),
        };

        let end_expr = match &self.ends {
            SliceParam::Static(ends) => ends[0].to_tokens(),
            SliceParam::Runtime(end_type) => self.get_scalar_expr(end_type),
        };

        (start_expr, end_expr)
    }

    fn generate_shape_slice(
        &self,
        shape: &crate::burn::ShapeType,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let shape_name = &shape.name;

        // Get the output rank from the output type
        let output_rank = match &self.output {
            Type::Shape(output_shape) => output_shape.rank,
            _ => panic!("Expected Shape output type for shape slice operation"),
        };
        let output_rank_lit = Literal::usize_unsuffixed(output_rank);

        match (&self.starts, &self.ends) {
            (SliceParam::Static(starts), SliceParam::Static(ends)) if starts.len() == 1 => {
                let start_val = starts[0];
                let end_val = ends[0];

                // Get step value if provided
                let step_val = if let Some(SliceParam::Static(ref steps)) = self.steps {
                    steps.first().copied().unwrap_or(1)
                } else {
                    1
                };

                // Always clamp start/end values
                let shape_len = shape.rank as i64;

                // Handle negative indices and clamp
                let actual_start = if start_val < 0 {
                    (shape_len + start_val).max(0) as usize
                } else {
                    start_val.min(shape_len) as usize
                };

                let actual_end = if end_val == i64::MAX {
                    shape.rank
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
                // Check if we have 1D tensor inputs (not supported for shape slicing)
                if let (
                    SliceParam::Runtime(Type::Tensor(start_t)),
                    SliceParam::Runtime(Type::Tensor(end_t)),
                ) = (&self.starts, &self.ends)
                    && start_t.rank == 1
                    && end_t.rank == 1
                {
                    panic!(
                        "1D tensor slicing is not supported for shape inputs - shapes must be sliced with scalar or static indices"
                    );
                }

                // Runtime slicing with scalars
                let (start_expr, end_expr) = self.get_slice_range_expressions();
                let shape_len_lit = Literal::i64_suffixed(shape.rank as i64);

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

    fn get_scalar_expr(&self, scalar_type: &Type) -> TokenStream {
        match scalar_type {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            Type::Shape(shape) => {
                let name = &shape.name;
                // For single-dimension slicing, use the first element of the shape
                quote! { #name[0] }
            }
            Type::Tensor(tensor) if tensor.rank == 1 => {
                // For 1D tensor, we'll handle it specially in the calling code
                // This shouldn't be called for 1D tensors as they use different logic
                panic!(
                    "1D tensor slice parameters should be handled separately, not through get_scalar_expr"
                )
            }
            _ => panic!(
                "Expected scalar, shape, or 1D tensor type for runtime slice parameter, got {scalar_type:?}"
            ),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        let mut inputs = vec![self.input.clone()];

        // Add runtime inputs if needed
        if let SliceParam::Runtime(ref start_type) = self.starts {
            inputs.push(start_type.clone());
        }
        if let SliceParam::Runtime(ref end_type) = self.ends {
            inputs.push(end_type.clone());
        }

        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        match &self.input {
            Type::Tensor(tensor) => {
                self.generate_tensor_slice(tensor, scope, node_position, output)
            }
            Type::Shape(shape) => self.generate_shape_slice(shape, output),
            _ => panic!("Unsupported input type for SliceNode"),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        super::Node::Slice(self)
    }
}

impl OnnxIntoNode for SliceNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = Type::from(node.inputs.first().unwrap());
        let output = Type::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::slice::SliceConfig>();
        use onnx_ir::node::slice::SliceInput;

        // Convert starts parameter
        let starts_param = match &config.starts {
            SliceInput::Static(values) => SliceParam::Static(values.clone()),
            SliceInput::Runtime(starts_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let starts_arg = &node.inputs[starts_ref.input_index];
                SliceParam::Runtime(Type::from(starts_arg))
            }
        };

        // Convert ends parameter
        let ends_param = match &config.ends {
            SliceInput::Static(values) => SliceParam::Static(values.clone()),
            SliceInput::Runtime(ends_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let ends_arg = &node.inputs[ends_ref.input_index];
                SliceParam::Runtime(Type::from(ends_arg))
            }
        };

        let mut slice_node = Self::new(input, output, starts_param, ends_param);

        // Convert axes parameter if present
        if let Some(ref axes) = config.axes {
            let axes_param = match axes {
                SliceInput::Static(values) => SliceParam::Static(values.clone()),
                SliceInput::Runtime(axes_ref) => {
                    // Get the actual argument using the RuntimeInputRef
                    let axes_arg = &node.inputs[axes_ref.input_index];
                    SliceParam::Runtime(Type::from(axes_arg))
                }
            };
            slice_node = slice_node.with_axes(axes_param);
        }

        // Convert steps parameter if present
        if let Some(ref steps) = config.steps {
            let steps_param = match steps {
                SliceInput::Static(values) => SliceParam::Static(values.clone()),
                SliceInput::Runtime(steps_ref) => {
                    // Get the actual argument using the RuntimeInputRef
                    let steps_arg = &node.inputs[steps_ref.input_index];
                    SliceParam::Runtime(Type::from(steps_arg))
                }
            };
            slice_node = slice_node.with_steps(steps_param);
        }

        slice_node
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{ShapeType, TensorType, graph::BurnGraph, node::test::assert_tokens};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_slice_tensor_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Static(vec![0, 1, 2]),
            SliceParam::Static(vec![3, 4, 5]),
        ));
        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 3> {
                    let tensor2 = tensor1.slice(s![0..3, 1..4, 2..5]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_tensor_runtime_scalars() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "start",
                crate::burn::ScalarKind::Int64,
            ))),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            ))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "start".to_string(),
                "end".to_string(),
            ],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, start: i64, end: i64) -> Tensor<B, 2> {
                    let tensor2 = tensor1.slice(s![start..end, ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            SliceParam::Static(vec![1]),
            SliceParam::Static(vec![3]),
        ));
        graph.register_input_output(
            vec!["shape1".to_string()],
            vec!["shape2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, shape1: [i64; 4]) -> [i64; 2] {
                    let shape2: [i64; 2] = shape1[1..3].try_into().unwrap();
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_runtime() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "start",
                crate::burn::ScalarKind::Int64,
            ))),
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            ))),
        ));
        graph.register_input_output(
            vec!["shape1".to_string(), "start".to_string(), "end".to_string()],
            vec!["shape2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, shape1: [i64; 4], start: i64, end: i64) -> [i64; 2] {
                    let start_val = start as i64;
                    let end_val = end as i64;
                    let start_idx = if start_val < 0 { (4i64 + start_val) as usize } else { start_val as usize };
                    let end_idx = if end_val < 0 { (4i64 + end_val) as usize } else { end_val as usize };
                    let shape2: [i64; 2] = shape1[start_idx..end_idx].try_into().unwrap();
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_tensor_runtime_shapes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Runtime(Type::Shape(ShapeType::new("start_shape", 1))),
            SliceParam::Runtime(Type::Shape(ShapeType::new("end_shape", 1))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "start_shape".to_string(),
                "end_shape".to_string(),
            ],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 3>, start_shape: [i64; 1], end_shape: [i64; 1]) -> Tensor<B, 3> {
                    let tensor2 = tensor1.slice(s![start_shape[0]..end_shape[0], .., ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_1d_tensor_params() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("starts", 1))),
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("ends", 1))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "starts".to_string(),
                "ends".to_string(),
            ],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 3>, starts: Tensor<B, 1, Int>, ends: Tensor<B, 1, Int>) -> Tensor<B, 3> {
                    let input_dims = tensor1.dims();
                    let start_data = starts.to_data();
                    let start_vec: alloc::vec::Vec<i64> = start_data.iter::<i64>().collect();
                    let end_data = ends.to_data();
                    let end_vec: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    let tensor2 = tensor1.slice(s![
                        start_vec.get(0).map(|&s| s as usize).unwrap_or(0)..end_vec.get(0).map(|&e| e as usize).unwrap_or(input_dims[0]),
                        start_vec.get(1).map(|&s| s as usize).unwrap_or(0)..end_vec.get(1).map(|&e| e as usize).unwrap_or(input_dims[1]),
                        start_vec.get(2).map(|&s| s as usize).unwrap_or(0)..end_vec.get(2).map(|&e| e as usize).unwrap_or(input_dims[2])
                    ]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_mixed_1d_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            SliceParam::Static(vec![0, 1]), // Static start
            SliceParam::Runtime(Type::Tensor(TensorType::new_int("ends", 1))), // 1D tensor end
        ));
        graph.register_input_output(
            vec!["tensor1".to_string(), "ends".to_string()],
            vec!["tensor2".to_string()],
            &[],
            &[],
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, ends: Tensor<B, 1, Int>) -> Tensor<B, 2> {
                    let input_dims = tensor1.dims();
                    let end_data = ends.to_data();
                    let end_vec: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    let tensor2 = tensor1.slice(s![
                        0i64 as usize..end_vec.get(0).map(|&e| e as usize).unwrap_or(input_dims[0]),
                        1i64 as usize..end_vec.get(1).map(|&e| e as usize).unwrap_or(input_dims[1])
                    ]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
