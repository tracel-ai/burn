use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::unsqueeze::UnsqueezeConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct UnsqueezeNode {
    pub input: Type,
    pub output: Type,
    pub axes: UnsqueezeConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnsqueezeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        let input = self.input.clone();
        match &self.axes {
            UnsqueezeConfig::Static(_) => vec![input],
            UnsqueezeConfig::Runtime(rt_type) => vec![input, Type::from(rt_type)],
        }
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let axes = match &self.axes {
            UnsqueezeConfig::Static(static_axes) => static_axes.to_tokens(),
            UnsqueezeConfig::Runtime(arg) => match Type::from(arg) {
                Type::Tensor(axes_tensor) => {
                    let tensor_name = &axes_tensor.name;
                    quote! {
                        #tensor_name.to_data().as_slice::<B::IntElem>().unwrap().iter().map(|&x| x.to_isize()).collect::<Vec<isize>>()
                    }
                }
                _ => panic!(
                    "UnsqueezeNode received invalid axes type: expected tensor but got {arg:?}"
                ),
            },
        };

        match (&self.input, &self.output) {
            (Type::Tensor(input), Type::Tensor(output)) => {
                let input = scope.tensor_use_owned(input, node_position);
                let output_name = &output.name;
                let output_rank = output.rank.to_tokens();
                quote! {
                    let #output_name: Tensor<B, #output_rank> = #input.unsqueeze_dims(&#axes);
                }
            }
            (Type::Scalar(scalar), Type::Tensor(output)) => {
                let scalar_name = &scalar.name;
                let output_name = &output.name;
                let output_rank = output.rank.to_tokens();

                // Determine the element type based on the output tensor type
                let elem_conversion = match &output.kind {
                    crate::burn::TensorKind::Int => quote! { #scalar_name.elem::<B::IntElem>() },
                    crate::burn::TensorKind::Float => {
                        quote! { #scalar_name.elem::<B::FloatElem>() }
                    }
                    crate::burn::TensorKind::Bool => quote! { #scalar_name != 0 },
                };

                // Generate the tensor creation code with appropriate type
                match &output.kind {
                    crate::burn::TensorKind::Int => quote! {
                        let #output_name = Tensor::<B, #output_rank, Int>::from_data([#elem_conversion], &self.device).unsqueeze();
                    },
                    crate::burn::TensorKind::Float => quote! {
                        let #output_name = Tensor::<B, #output_rank>::from_data([#elem_conversion], &self.device).unsqueeze();
                    },
                    crate::burn::TensorKind::Bool => quote! {
                        let #output_name = Tensor::<B, #output_rank, Bool>::from_data([#elem_conversion], &self.device).unsqueeze();
                    },
                }
            }
            (Type::Scalar(scalar), Type::Shape(shape)) => {
                // Scalar(Int) -> Shape[1] conversion: Reverses squeeze(Shape[1]) -> Scalar
                // Common in ONNX for dynamic reshape operations where dimensions are computed at runtime.
                // This is a zero-cost conversion (both types are CPU-resident) that avoids unnecessary
                // tensor allocations and GPU transfers.
                let input_name = &scalar.name;
                let output_name = &shape.name;

                // Only Int32/Int64 scalars can be converted to Shape
                let value_expr = match &scalar.kind {
                    crate::burn::ScalarKind::Int64 => quote! { #input_name },
                    crate::burn::ScalarKind::Int32 => quote! { #input_name as i64 },
                    _ => panic!(
                        "Unsqueeze from Scalar to Shape only supports Int32/Int64 input types, but got: {:?}",
                        scalar.kind
                    ),
                };

                // Create a shape array with the scalar value
                // For unsqueeze on axis 0, we're creating a 1D shape from a scalar
                quote! {
                    let #output_name = [#value_expr];
                }
            }
            _ => panic!(
                "UnsqueezeNode received unsupported input/output combination: {:?} -> {:?}",
                self.input, self.output
            ),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unsqueeze(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.axes {
            UnsqueezeConfig::Runtime(_) => {
                imports.register("alloc::vec::Vec");
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
        ScalarKind, ScalarType, ShapeType, TensorType, Type,
        graph::BurnGraph,
        node::{test::assert_tokens, unsqueeze::UnsqueezeNode},
    };

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(UnsqueezeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            Type::Tensor(TensorType::new_float("tensor2", 5)),
            UnsqueezeConfig::Static([0, 4].into()),
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
                pub fn forward(&self, tensor1: Tensor<B, 3>) -> Tensor<B, 5> {
                    let tensor2: Tensor<B, 5> = tensor1.unsqueeze_dims(&[0,4]);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_unsqueeze_scalar_to_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(UnsqueezeNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int64)),
            Type::Shape(ShapeType::new("shape1", 1)),
            UnsqueezeConfig::Static([0].into()),
        ));

        graph.register_input_output(vec!["scalar1".to_string()], vec!["shape1".to_string()]);

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
                pub fn forward(&self, scalar1: i64) -> [i64; 1] {
                    let shape1 = [scalar1];
                    shape1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_unsqueeze_int32_scalar_to_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(UnsqueezeNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int32)),
            Type::Shape(ShapeType::new("shape1", 1)),
            UnsqueezeConfig::Static([0].into()),
        ));

        graph.register_input_output(vec!["scalar1".to_string()], vec!["shape1".to_string()]);

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
                pub fn forward(&self, scalar1: i32) -> [i64; 1] {
                    let shape1 = [scalar1 as i64];
                    shape1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
