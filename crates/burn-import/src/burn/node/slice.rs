use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

#[derive(Debug, Clone)]
pub struct SliceNode {
    pub input: Type,
    pub output: Type,
    pub config: SliceNodeConfig,
}

#[derive(Debug, Clone)]
pub enum SliceNodeConfig {
    Static { ranges: Vec<Option<(i64, i64)>> },
    Runtime { starts: Type, ends: Type },
    Mixed { starts: SliceParam, ends: SliceParam },
}

#[derive(Debug, Clone)]
pub enum SliceParam {
    Static(Vec<i64>),
    Runtime(Type),
}

impl SliceNode {
    pub fn new_static(input: Type, output: Type, ranges: Vec<Option<(i64, i64)>>) -> Self {
        Self {
            input,
            output,
            config: SliceNodeConfig::Static { ranges },
        }
    }

    pub fn new_runtime(input: Type, output: Type, starts: Type, ends: Type) -> Self {
        Self {
            input,
            output,
            config: SliceNodeConfig::Runtime { starts, ends },
        }
    }

    pub fn new_mixed(input: Type, output: Type, starts: SliceParam, ends: SliceParam) -> Self {
        Self {
            input,
            output,
            config: SliceNodeConfig::Mixed { starts, ends },
        }
    }

    fn generate_static_slice(
        &self,
        ranges: &[Option<(i64, i64)>],
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        let output_rank = match self.output {
            Type::Tensor(ref tensor) => tensor.rank,
            Type::Shape(ref shape) => shape.rank,
            _ => panic!("Invalid output type"),
        };
        let output_rank = Literal::usize_unsuffixed(output_rank);

        let ranges = ranges.iter().map(|range| match range {
            Some((start, end)) => {
                let start = start.to_tokens();
                let end = end.to_tokens();
                quote! { #start..#end}
            }
            None => quote! { .. },
        });

        match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);
                quote! {
                    let #output = #input.slice(s![#(#ranges),*]);
                }
            }
            Type::Shape(shape) => {
                let shape_len = shape.rank;
                let shape_len = Literal::usize_unsuffixed(shape_len);
                let shape = shape.name.clone();

                let mut ranges = ranges;
                let range = ranges.next().unwrap();

                quote! {
                    let #output: [usize; #output_rank] = #shape[s![#range].into_ranges([#shape_len].into())[0].clone()].try_into().unwrap();
                }
            }
            _ => {
                panic!("Unsupported input type for SliceNode");
            }
        }
    }

    fn generate_runtime_slice(
        &self,
        starts: &Type,
        ends: &Type,
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);

                let starts_var = match starts {
                    Type::Scalar(scalar) => {
                        let name = scalar.name.clone();
                        quote! { #name as usize }
                    }
                    _ => panic!("starts must be a scalar for runtime slice"),
                };

                let ends_var = match ends {
                    Type::Scalar(scalar) => {
                        let name = scalar.name.clone();
                        quote! { #name as usize }
                    }
                    _ => panic!("ends must be a scalar for runtime slice"),
                };

                quote! {
                    let #output = #input.slice(s![#starts_var..#ends_var, ..]);
                }
            }
            Type::Shape(_) => {
                panic!("Runtime slice on Shape input is not supported");
            }
            _ => {
                panic!("Unsupported input type for runtime SliceNode");
            }
        }
    }

    fn generate_mixed_slice(
        &self,
        starts: &SliceParam,
        ends: &SliceParam,
        scope: &mut Scope,
        node_position: usize,
        output: &proc_macro2::Ident,
    ) -> TokenStream {
        match &self.input {
            Type::Tensor(tensor) => {
                let input = scope.tensor_use_owned(tensor, node_position);

                let starts_expr = match starts {
                    SliceParam::Static(values) => {
                        if values.len() != 1 {
                            panic!("Mixed slice currently supports only single dimension");
                        }
                        let start = values[0].to_tokens();
                        quote! { #start }
                    }
                    SliceParam::Runtime(scalar_type) => {
                        let name = match scalar_type {
                            Type::Scalar(scalar) => scalar.name.clone(),
                            _ => panic!("Runtime starts must be a scalar"),
                        };
                        quote! { #name as usize }
                    }
                };

                let ends_expr = match ends {
                    SliceParam::Static(values) => {
                        if values.len() != 1 {
                            panic!("Mixed slice currently supports only single dimension");
                        }
                        let end = values[0].to_tokens();
                        quote! { #end }
                    }
                    SliceParam::Runtime(scalar_type) => {
                        let name = match scalar_type {
                            Type::Scalar(scalar) => scalar.name.clone(),
                            _ => panic!("Runtime ends must be a scalar"),
                        };
                        quote! { #name as usize }
                    }
                };

                quote! {
                    let #output = #input.slice(s![#starts_expr..#ends_expr, ..]);
                }
            }
            Type::Shape(_) => {
                panic!("Mixed slice on Shape input is not supported");
            }
            _ => {
                panic!("Unsupported input type for mixed SliceNode");
            }
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }
    fn input_types(&self) -> Vec<Type> {
        match &self.config {
            SliceNodeConfig::Static { .. } => vec![self.input.clone()],
            SliceNodeConfig::Runtime { starts, ends } => {
                vec![self.input.clone(), starts.clone(), ends.clone()]
            }
            SliceNodeConfig::Mixed { starts, ends } => {
                let mut inputs = vec![self.input.clone()];
                if let SliceParam::Runtime(start_type) = starts {
                    inputs.push(start_type.clone());
                }
                if let SliceParam::Runtime(end_type) = ends {
                    inputs.push(end_type.clone());
                }
                inputs
            }
        }
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        match &self.config {
            SliceNodeConfig::Static { ranges } => {
                self.generate_static_slice(ranges, scope, node_position, output)
            }
            SliceNodeConfig::Runtime { starts, ends } => {
                self.generate_runtime_slice(starts, ends, scope, node_position, output)
            }
            SliceNodeConfig::Mixed { starts, ends } => {
                self.generate_mixed_slice(starts, ends, scope, node_position, output)
            }
        }
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
        graph.register(SliceNode::new_static(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            vec![Some((0, 1)), None, None, None],
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
        graph.register(SliceNode::new_static(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 2)),
            vec![Some((1, 3))],
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
        graph.register(SliceNode::new_static(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Shape(ShapeType::new("shape2", 4)),
            vec![None],
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
                     let shape2: [usize; 4] = shape1[s![..].into_ranges([4].into())[0].clone()].try_into().unwrap();
                     shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_slice_runtime() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(SliceNode::new_runtime(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Scalar(crate::burn::ScalarType::new(
                "start",
                crate::burn::ScalarKind::Int64,
            )),
            Type::Scalar(crate::burn::ScalarType::new(
                "end",
                crate::burn::ScalarKind::Int64,
            )),
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
        graph.register(SliceNode::new_mixed(
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
