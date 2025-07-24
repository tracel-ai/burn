use super::{Node, NodeCodegen};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SqueezeNode {
    pub input: Type,
    pub output: Type,
    pub axes: Vec<i64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match (&self.input, &self.output) {
            (Type::Tensor(input), Type::Tensor(output)) => {
                let input = scope.tensor_use_owned(input, node_position);
                let output = &output.name;
                let axes_arg = &self.axes.to_tokens();

                quote! {
                    let #output = #input.squeeze_dims(&#axes_arg);
                }
            }
            (Type::Shape(input), Type::Scalar(output)) => {
                // Shape(1) squeezed on axis 0 produces a scalar
                let input_name = &input.name;
                let output_name = &output.name;

                // Cast to the appropriate scalar type
                let cast_expr = match &output.kind {
                    crate::burn::ScalarKind::Int64 => quote! { #input_name[0] as i64 },
                    crate::burn::ScalarKind::Int32 => quote! { #input_name[0] as i32 },
                    _ => panic!(
                        "Squeeze from Shape to Scalar only supports Int32/Int64 output types"
                    ),
                };

                quote! {
                    let #output_name = #cast_expr;
                }
            }
            (Type::Shape(input), Type::Shape(output)) => {
                // Shape(n) where n > 1 remains unchanged (squeeze is a no-op)
                let input_name = &input.name;
                let output_name = &output.name;

                quote! {
                    let #output_name = #input_name;
                }
            }
            _ => panic!(
                "Squeeze: unsupported input/output combination: {:?} -> {:?}",
                self.input, self.output
            ),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Squeeze(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarType, ShapeType, TensorType, Type,
        graph::BurnGraph,
        node::{squeeze::SqueezeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SqueezeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            [1].into(),
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
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 2> {
                    let tensor2 = tensor1.squeeze_dims(&[1]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_squeeze_shape_to_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SqueezeNode::new(
            Type::Shape(ShapeType::new("shape1", 1)),
            Type::Scalar(ScalarType::new(
                "scalar1",
                crate::burn::ty::ScalarKind::Int64,
            )),
            [0].into(),
        ));

        graph.register_input_output(vec!["shape1".to_string()], vec!["scalar1".to_string()]);

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
                pub fn forward(&self, shape1: [usize; 1]) -> i64 {
                    let scalar1 = shape1[0] as i64;
                    scalar1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_squeeze_shape_no_op() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SqueezeNode::new(
            Type::Shape(ShapeType::new("shape1", 2)),
            Type::Shape(ShapeType::new("shape2", 2)),
            [0].into(),
        ));

        graph.register_input_output(vec!["shape1".to_string()], vec!["shape2".to_string()]);

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
                pub fn forward(&self, shape1: [usize; 2]) -> [usize; 2] {
                    let shape2 = shape1;
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
