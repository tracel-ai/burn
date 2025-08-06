use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub enum ReshapeShape {
    Static(Vec<i64>),
    Runtime(Type),
}

#[derive(Debug, Clone)]
pub struct ReshapeNode {
    pub input: TensorType,
    pub output: TensorType,
    pub shape: ReshapeShape,
}

impl ReshapeNode {
    pub fn new<S: Into<ReshapeShape>>(input: TensorType, output: TensorType, shape: S) -> Self {
        Self {
            input,
            output,
            shape: shape.into(),
        }
    }
}

impl From<Vec<i64>> for ReshapeShape {
    fn from(shape: Vec<i64>) -> Self {
        ReshapeShape::Static(shape)
    }
}

impl From<Type> for ReshapeShape {
    fn from(shape: Type) -> Self {
        ReshapeShape::Runtime(shape)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReshapeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        match &self.shape {
            ReshapeShape::Static(_) => vec![Type::Tensor(self.input.clone())],
            ReshapeShape::Runtime(shape_type) => {
                vec![Type::Tensor(self.input.clone()), shape_type.clone()]
            }
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        match &self.shape {
            ReshapeShape::Static(shape_values) => {
                let shape_values = shape_values.to_tokens();
                quote! {
                    let #output = #input.reshape(#shape_values);
                }
            }
            ReshapeShape::Runtime(shape_type) => {
                match shape_type {
                    Type::Shape(shape) => {
                        let shape_name = &shape.name;
                        quote! {
                            let #output = #input.reshape(#shape_name);
                        }
                    }
                    Type::Tensor(tensor) => {
                        let shape_name = &tensor.name;
                        let output_rank = self.output.rank;

                        // Generate a const generic array initialization
                        // This will create: shape_array[0] as usize, shape_array[1] as usize, ...
                        let array_init = (0..output_rank)
                            .map(|i| {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                quote! { shape_array[#idx] as usize }
                            })
                            .collect::<Vec<_>>();

                        quote! {
                            let shape_data = #shape_name.to_data();
                            let shape_array = shape_data.as_slice::<i64>().unwrap();
                            let #output = #input.reshape([#(#array_init),*]);
                        }
                    }
                    _ => panic!("Shape input must be a tensor or shape type"),
                }
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Reshape(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{reshape::ReshapeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_reshape_with_static_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            vec![4, 4, 4, 4],
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
                    let tensor2 = tensor1.reshape([4, 4, 4, 4]);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reshape_with_tensor_as_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            TensorType::new_float("tensor1", 1),
            TensorType::new_float("output", 2),
            Type::Tensor(TensorType::new_int("shape", 1)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 1>, shape: Tensor<B, 1, Int>) -> Tensor<B, 2> {
                    let shape_data = shape.to_data();
                    let shape_array = shape_data.as_slice::<i64>().unwrap();
                    let output = tensor1.reshape([shape_array[0] as usize, shape_array[1] as usize]);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reshape_with_shape_type() {
        use crate::burn::ShapeType;

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("output", 2),
            Type::Shape(ShapeType::new("shape", 2)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 4>, shape: [i64; 2]) -> Tensor<B, 2> {
                    let output = tensor1.reshape(shape);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
