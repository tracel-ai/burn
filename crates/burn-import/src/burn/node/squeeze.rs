use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SqueezeNode {
    pub input: Type,
    pub output: Type,
    pub axes: Option<Vec<i64>>,
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
                let input_tensor = scope.tensor_use_owned(input, node_position);
                let output_name = &output.name;

                match &self.axes {
                    Some(axes_vec) => {
                        // Use squeeze_dims with specific axes
                        let axes_arg = axes_vec.to_tokens();
                        quote! {
                            let #output_name = #input_tensor.squeeze_dims(&#axes_arg);
                        }
                    }
                    None => {
                        // When axes is None, squeeze all dimensions with size 1
                        let output_rank = output.rank;
                        quote! {
                            let #output_name = #input_tensor.squeeze::<#output_rank>();
                        }
                    }
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
            (Type::Scalar(input), Type::Scalar(output)) => {
                // Scalar squeeze is a no-op
                let input_name = &input.name;
                let output_name = &output.name;

                quote! {
                    let #output_name = #input_name;
                }
            }
            (Type::Tensor(input), Type::Scalar(output)) => {
                // This handles ONNX models where single-element tensors need to be converted to scalars
                // Works for all tensor types (Float, Int, Bool) using the .into_scalar() method
                let input = scope.tensor_use_owned(input, node_position);
                let output_name = &output.name;

                // Use .into_scalar() and cast to the appropriate concrete type using .elem::<T>()
                let elem_cast = match &output.kind {
                    crate::burn::ScalarKind::Float32 => quote! { .elem::<f32>() },
                    crate::burn::ScalarKind::Float64 => quote! { .elem::<f64>() },
                    crate::burn::ScalarKind::Int32 => quote! { .elem::<i32>() },
                    crate::burn::ScalarKind::Int64 => quote! { .elem::<i64>() },
                    crate::burn::ScalarKind::Bool => quote! { .elem::<bool>() },
                };

                quote! {
                    let #output_name = #input.into_scalar()#elem_cast;
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

impl OnnxIntoNode for SqueezeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Squeeze(n) = &node else {
            panic!("Expected Squeeze node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let input = Type::from(inputs.first().unwrap());
        let output = Type::from(outputs.first().unwrap());
        let axes = config.axes.as_ref().map(|a| match a {
            onnx_ir::node::squeeze::SqueezeInput::Static(axes) => axes.clone(),
            onnx_ir::node::squeeze::SqueezeInput::Runtime(_) => {
                panic!("Runtime squeeze axes not yet supported in burn-import")
            }
        });
        Self::new(input, output, axes)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, ShapeType, TensorType, Type,
        graph::BurnGraph,
        node::{squeeze::SqueezeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SqueezeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Some(vec![1]),
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
            Some(vec![0]),
        ));

        graph.register_input_output(
            vec!["shape1".to_string()],
            vec!["scalar1".to_string()],
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
                pub fn forward(&self, shape1: [i64; 1]) -> i64 {
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
            Some(vec![0]),
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
                pub fn forward(&self, shape1: [i64; 2]) -> [i64; 2] {
                    let shape2 = shape1;
                    shape2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_squeeze_scalar_no_op() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SqueezeNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float32)),
            None,
        ));

        graph.register_input_output(
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
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
                pub fn forward(&self, scalar1: f32) -> f32 {
                    let scalar2 = scalar1;
                    scalar2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
