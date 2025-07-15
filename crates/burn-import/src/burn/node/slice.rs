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

        match (&self.starts, &self.ends) {
            (SliceParam::Static(starts), SliceParam::Static(ends)) if starts.len() == 1 => {
                // Simple static case
                let start = starts[0].to_tokens();
                let end = ends[0].to_tokens();
                let rank = tensor.rank;
                let mut ranges = vec![quote! { .. }; rank];
                ranges[0] = quote! { #start..#end };

                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            (SliceParam::Runtime(start_type), SliceParam::Runtime(end_type)) => {
                // Both runtime
                let start_expr = self.get_scalar_expr(start_type);
                let end_expr = self.get_scalar_expr(end_type);
                let rank = tensor.rank;
                let mut ranges = vec![quote! { .. }; rank];
                ranges[0] = quote! { #start_expr..#end_expr };

                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            (SliceParam::Static(starts), SliceParam::Runtime(end_type)) if starts.len() == 1 => {
                // Static start, runtime end
                let start = starts[0].to_tokens();
                let end_expr = self.get_scalar_expr(end_type);
                let rank = tensor.rank;
                let mut ranges = vec![quote! { .. }; rank];
                ranges[0] = quote! { #start..#end_expr };

                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            (SliceParam::Runtime(start_type), SliceParam::Static(ends)) if ends.len() == 1 => {
                // Runtime start, static end
                let start_expr = self.get_scalar_expr(start_type);
                let end = ends[0].to_tokens();
                let rank = tensor.rank;
                let mut ranges = vec![quote! { .. }; rank];
                ranges[0] = quote! { #start_expr..#end };

                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            (SliceParam::Static(starts), SliceParam::Static(ends)) => {
                // Complex static case with multiple dimensions
                let rank = tensor.rank;
                let mut ranges = vec![quote! { .. }; rank];

                // Assuming axes are sequential from 0
                for i in 0..starts.len().min(ends.len()) {
                    let start = starts[i].to_tokens();
                    let end = ends[i].to_tokens();
                    ranges[i] = quote! { #start..#end };
                }

                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            _ => panic!("Unsupported slice configuration"),
        }
    }

    fn generate_shape_slice(
        &self,
        shape: &crate::burn::ShapeType,
        _scope: &mut Scope,
        _node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        // Shape slicing only supports static slicing
        match (&self.starts, &self.ends) {
            (SliceParam::Static(starts), SliceParam::Static(ends)) if starts.len() == 1 => {
                let start = starts[0].to_tokens();
                let end = ends[0].to_tokens();
                let shape_name = &shape.name;
                let shape_len = Literal::usize_unsuffixed(shape.rank);
                let output_len = (ends[0] - starts[0]) as usize;
                let output_rank = Literal::usize_unsuffixed(output_len);

                quote! {
                    let #output: [usize; #output_rank] = #shape_name[s![#start..#end].into_ranges([#shape_len].into())[0].clone()].try_into().unwrap();
                }
            }
            _ => panic!("Shape slicing only supports static slicing"),
        }
    }

    fn get_scalar_expr(&self, scalar_type: &Type) -> TokenStream {
        match scalar_type {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name as usize }
            }
            _ => panic!("Expected scalar type for runtime slice parameter"),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }
    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![self.input.clone()];

        match &self.starts {
            SliceParam::Runtime(start_type) => inputs.push(start_type.clone()),
            _ => {}
        }
        match &self.ends {
            SliceParam::Runtime(end_type) => inputs.push(end_type.clone()),
            _ => {}
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
        match &self.input {
            Type::Shape(_) => {
                imports.register("burn::tensor::RangesArg");
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
}
