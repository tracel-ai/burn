use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

#[derive(Debug, Clone)]
pub struct SliceNode {
    pub input: Type,
    pub output: Type,
    pub starts: SliceParam,
    pub ends: SliceParam,
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
        }
    }

    fn generate_slice(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        match &self.input {
            Type::Tensor(tensor) => {
                self.generate_tensor_slice(tensor, scope, node_position, output)
            }
            Type::Shape(shape) => self.generate_shape_slice(shape, scope, node_position, output),
            _ => panic!("Unsupported input type for SliceNode"),
        }
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
                let limit = starts.len().min(ends.len()).min(rank);
                for (i, range) in ranges.iter_mut().enumerate().take(limit) {
                    let start = starts[i].to_tokens();
                    let end = ends[i].to_tokens();
                    *range = quote! { #start..#end };
                }
            }

            // Both runtime: check if they're shapes for multi-dimensional slicing
            (SliceParam::Runtime(start_type), SliceParam::Runtime(end_type)) => {
                match (start_type, end_type) {
                    (Type::Shape(start_shape), Type::Shape(end_shape)) => {
                        let start_name = &start_shape.name;
                        let end_name = &end_shape.name;
                        let num_dims = start_shape.rank.min(end_shape.rank).min(rank);

                        for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            *range = quote! { #start_name[#idx]..#end_name[#idx] };
                        }
                    }
                    _ => {
                        // Single dimension slicing for scalars
                        let (start_expr, end_expr) = self.get_slice_range_expressions();
                        ranges[0] = quote! { #start_expr..#end_expr };
                    }
                }
            }

            // Static start, runtime end
            (SliceParam::Static(starts), SliceParam::Runtime(end_type)) => match end_type {
                Type::Shape(end_shape) => {
                    let end_name = &end_shape.name;
                    let num_dims = starts.len().min(end_shape.rank).min(rank);

                    for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                        let start = starts[i].to_tokens();
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        *range = quote! { #start..#end_name[#idx] };
                    }
                }
                _ => {
                    let (start_expr, end_expr) = self.get_slice_range_expressions();
                    ranges[0] = quote! { #start_expr..#end_expr };
                }
            },

            // Runtime start, static end
            (SliceParam::Runtime(start_type), SliceParam::Static(ends)) => match start_type {
                Type::Shape(start_shape) => {
                    let start_name = &start_shape.name;
                    let num_dims = start_shape.rank.min(ends.len()).min(rank);

                    for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        let end = ends[i].to_tokens();
                        *range = quote! { #start_name[#idx]..#end };
                    }
                }
                _ => {
                    let (start_expr, end_expr) = self.get_slice_range_expressions();
                    ranges[0] = quote! { #start_expr..#end_expr };
                }
            },
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
        _scope: &mut Scope,
        _node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let shape_name = &shape.name;
        let shape_len = Literal::usize_unsuffixed(shape.rank);

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

                // Handle special case: slice[-1:] pattern
                if start_val == -1 && (end_val == i64::MAX || end_val >= shape.rank as i64) {
                    // This gets the last element
                    quote! {
                        let #output: [usize; 1] = [#shape_name[#shape_name.len() - 1]];
                    }
                } else if start_val < 0 || end_val < 0 {
                    // Handle negative indices - convert at compile time since we know the shape length
                    let shape_len = shape.rank as i64;
                    let actual_start = if start_val < 0 {
                        (shape_len + start_val).max(0) as usize
                    } else {
                        start_val as usize
                    };
                    let actual_end = if end_val < 0 {
                        (shape_len + end_val).max(0) as usize
                    } else {
                        end_val as usize
                    };

                    let start_lit = Literal::usize_unsuffixed(actual_start);
                    let end_lit = Literal::usize_unsuffixed(actual_end);

                    quote! {
                        let #output: [usize; #output_rank_lit] = #shape_name[#start_lit..#end_lit].try_into().unwrap();
                    }
                } else {
                    // Positive indices
                    let start = start_val.to_tokens();
                    let end = if end_val == i64::MAX {
                        quote! { #shape_len }
                    } else {
                        end_val.to_tokens()
                    };
                    let output_len = if end_val == i64::MAX {
                        shape.rank.saturating_sub(start_val as usize)
                    } else {
                        (end_val as usize).saturating_sub(start_val as usize)
                    };
                    let output_rank = Literal::usize_unsuffixed(output_len);

                    quote! {
                        let #output: [usize; #output_rank] = #shape_name[s![#start..#end].into_ranges([#shape_len].into())[0].clone()].try_into().unwrap();
                    }
                }
            }
            _ => {
                // Runtime slicing - we still know the output size from type inference
                // and we know the shape length at compile time
                let (start_expr, end_expr) = self.get_slice_range_expressions();
                let shape_len_lit = Literal::i64_unsuffixed(shape.rank as i64);

                quote! {
                    let _start_val = #start_expr as i64;
                    let _end_val = #end_expr as i64;
                    let _start = if _start_val < 0 { (#shape_len_lit + _start_val) as usize } else { _start_val as usize };
                    let _end = if _end_val < 0 { (#shape_len_lit + _end_val) as usize } else { _end_val as usize };
                    let #output: [usize; #output_rank_lit] = #shape_name[_start.._end].try_into().unwrap();
                }
            }
        }
    }

    fn get_scalar_expr(&self, scalar_type: &Type) -> TokenStream {
        match scalar_type {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name as usize }
            }
            Type::Shape(shape) => {
                let name = &shape.name;
                // For single-dimension slicing, use the first element of the shape
                quote! { #name[0] }
            }
            _ => panic!(
                "Expected scalar or shape type for runtime slice parameter, got {scalar_type:?}"
            ),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }
    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![self.input.clone()];

        if let SliceParam::Runtime(start_type) = &self.starts {
            inputs.push(start_type.clone());
        }
        if let SliceParam::Runtime(end_type) = &self.ends {
            inputs.push(end_type.clone());
        }

        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        self.generate_slice(scope, node_position)
    }
    fn into_node(self) -> Node<PS> {
        Node::Slice(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::s");
        if matches!(&self.input, Type::Shape(_)) {
            imports.register("burn::tensor::RangesArg");
        }
    }
}
#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ShapeType, TensorType,
        graph::BurnGraph,
        node::{slice::SliceNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_slice_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            SliceParam::Static(vec![0]),
            SliceParam::Static(vec![1]),
        ));
        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::s; // Added import
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
                    let tensor2 = tensor1.slice(s![0..1, .., .., ..]); // Use s! directly
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            SliceParam::Static(vec![1]),
            SliceParam::Static(vec![3]),
        ));
        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::RangesArg;
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
                pub fn forward(&self, shape1: [usize; 4]) -> [usize; 2] {
                     // Use s!, ensure no usize suffix
                     let shape2: [usize; 2] = shape1[s![1..3].into_ranges([4].into())[0].clone()].try_into().unwrap();
                     shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_full_range() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 4)),
            SliceParam::Static(vec![0]),
            SliceParam::Static(vec![4]),
        ));
        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

        let expected = quote! {
            use burn::tensor::s; // Added import
            use burn::tensor::RangesArg;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor}, // Removed Shape
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
                pub fn forward(&self, shape1: [usize; 4]) -> [usize; 4] {
                    // Use s!, ensure no usize suffix
                     let shape2: [usize; 4] = shape1[s![0..4].into_ranges([4].into())[0].clone()].try_into().unwrap();
                     shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_runtime() {
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
        );

        let expected = quote! {
            use burn::tensor::s;
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, start: i64, end: i64) -> Tensor<B, 2> {
                    let tensor2 = tensor1.slice(s![start as usize..end as usize, ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_mixed() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            SliceParam::Static(vec![1]), // Static start
            SliceParam::Runtime(Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            ))), // Runtime end
        ));
        graph.register_input_output(
            vec!["tensor1".to_string(), "end".to_string()],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
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
                pub fn forward(&self, tensor1: Tensor<B, 2>, end: i64) -> Tensor<B, 2> {
                    let tensor2 = tensor1.slice(s![1..end as usize, ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_shape_params() {
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
        );

        let expected = quote! {
            use burn::tensor::s;
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
                pub fn forward(&self, tensor1: Tensor<B, 3>, start_shape: [usize; 1], end_shape: [usize; 1]) -> Tensor<B, 3> {
                    let tensor2 = tensor1.slice(s![start_shape[0]..end_shape[0], .., ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_multi_dim_shape_params() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            SliceParam::Runtime(Type::Shape(ShapeType::new("start_shape", 3))),
            SliceParam::Runtime(Type::Shape(ShapeType::new("end_shape", 3))),
        ));
        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "start_shape".to_string(),
                "end_shape".to_string(),
            ],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::s;
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
                pub fn forward(&self, tensor1: Tensor<B, 4>, start_shape: [usize; 3], end_shape: [usize; 3]) -> Tensor<B, 4> {
                    let tensor2 = tensor1.slice(s![start_shape[0]..end_shape[0], start_shape[1]..end_shape[1], start_shape[2]..end_shape[2], ..]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
