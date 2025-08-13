use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SliceNode {
    pub input: Type,
    pub output: Type,
    pub ranges: Vec<Option<(i64, i64)>>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SliceNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();

        let output_rank = match self.output {
            Type::Tensor(ref tensor) => tensor.rank,
            Type::Shape(ref shape) => shape.rank,
            _ => panic!("Invalid output type"),
        };
        // Use unsuffixed literal to avoid output like `1usize`
        let output_rank = Literal::usize_unsuffixed(output_rank);

        let ranges = self.ranges.iter().map(|range| match range {
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
                let shape_len = Literal::usize_unsuffixed(shape_len); // Use unsuffixed literal
                let shape = shape.name.clone();

                // first item from the range
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
            vec![Some((0, 1)), None, None, None],
        ));
        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::s; // Added import
            use burn::tensor::Tensor;
            use burn::{
                module::Module,
                tensor::backend::Backend,
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
            vec![Some((1, 3))],
        ));
        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

        let expected = quote! {
            use burn::tensor::s;
            use burn::tensor::RangesArg;
            use burn::{
                module::Module,
                tensor::backend::Backend,
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
            vec![None],
        ));
        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

        let expected = quote! {
            use burn::tensor::s; // Added import
            use burn::tensor::RangesArg;
            use burn::{
                module::Module,
                tensor::backend::Backend, // Removed Shape
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
}
