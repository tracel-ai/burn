use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::expand::ExpandShape;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ExpandNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: ExpandShape,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ExpandNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        let input = Type::Tensor(self.input.clone());
        // If the shape is static, we only have the input tensor as an input,
        // if it is dynamic, the shape will be our 2nd:
        match &self.shape {
            ExpandShape::Static(_) => vec![input],
            ExpandShape::Runtime(rt_type) => vec![input, Type::from(rt_type)],
        }
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let output_rank = &self.output.rank;

        let shape = match &self.shape {
            ExpandShape::Static(static_shape) => static_shape.to_tokens(),
            ExpandShape::Runtime(ty) => match Type::from(ty) {
                Type::Tensor(shape_tensor) => {
                    let tensor_name = &shape_tensor.name;
                    quote! {
                        TryInto::<[B::IntElem; #output_rank]>::try_into(#tensor_name.to_data().as_slice::<B::IntElem>().unwrap()).unwrap()
                    }
                }
                Type::Shape(shape) => {
                    // Shape arrays are [i64; N] and expand now accepts them directly via Element trait
                    let shape_name = &shape.name;
                    quote! { #shape_name }
                }
                b => panic!("Invalid shape source {b:?}"),
            },
        };

        quote! {
            let #output = #input.expand(#shape);
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Expand(self)
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
        node::{expand::ExpandNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_expand_static() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ExpandNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ExpandShape::Static([4, 4, 4, 4].into()),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.expand([4,4,4,4]);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
    #[test]
    fn test_codegen_expand_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ExpandNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ExpandShape::Runtime(Argument {
                name: "shape1".to_string(),
                ty: ArgType::Shape(4),
                value: None,
                passed: false,
            }),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape1".to_string()],
            vec!["tensor2".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    shape1: [i64; 4],
                ) -> Tensor<B, 4> {
                    let tensor2 = tensor1.expand(shape1);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_expand_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ExpandNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ExpandShape::Runtime(Argument {
                name: "tensor3".to_string(),
                ty: ArgType::Tensor(onnx_ir::TensorType {
                    elem_type: ElementType::Int32,
                    rank: 1,
                    static_shape: None,
                }),
                value: None,
                passed: false,
            }),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor3".to_string()],
            vec!["tensor2".to_string()],
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor3: Tensor<B, 1, Int>,
                ) -> Tensor<B, 4> {
                    let tensor2 = tensor1.expand(
                        TryInto::<[B::IntElem; 4usize]>::try_into(tensor3.to_data().as_slice::<B::IntElem>().unwrap())
                            .unwrap(),
                    );
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
