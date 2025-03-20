use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ExpandNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: ExpandShape,
}

#[derive(Debug, Clone)]
pub enum ExpandShape {
    Static(Vec<i64>),
    Runtime(Type),
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
            ExpandShape::Runtime(rt_type) => vec![input, rt_type.clone()],
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        let shape = match &self.shape {
            ExpandShape::Static(static_shape) => static_shape.to_tokens(),
            ExpandShape::Runtime(Type::Tensor(shape_tensor)) => {
                // since we don't take ownership of the shape_tensor, we don't need `tensor_use_owned` here:
                let tensor_name = &shape_tensor.name;
                let dim = shape_tensor.shape.as_ref().unwrap()[0];
                // the shape of the tensor is already validated statically to be rank one when parsing the input
                // we'll need to download the Tensor from device to cpu for expand operation.
                // Also, we'll need to convert it to an array for conversion into BroadcastArgs
                quote! {
                    TryInto::<[B::IntElem; #dim]>::try_into(#tensor_name.to_data().as_slice::<B::IntElem>().unwrap()).unwrap()
                }
            }
            ExpandShape::Runtime(Type::Shape(shape)) => {
                // Shape implements BroadcastArgs, so it can be passed to expand directly
                let shape_name = &shape.name;
                quote! { #shape_name }
            }
            _ => panic!("Invalid shape source {:?}", self.shape),
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

    use super::*;
    use crate::burn::{
        ShapeType, TensorType,
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
            ExpandShape::Runtime(Type::Shape(ShapeType::new("shape1", 4))),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape1".to_string()],
            vec!["tensor2".to_string()],
        );

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    shape1: [usize; 4],
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

        let mut shape_tensor_type = TensorType::new_int("tensor3", 4);
        shape_tensor_type.shape = Some(vec![4]);

        graph.register(ExpandNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            ExpandShape::Runtime(Type::Tensor(shape_tensor_type)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor3".to_string()],
            vec!["tensor2".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor3: Tensor<B, 4, Int>,
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
