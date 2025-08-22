use super::{Node, NodeCodegen};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use onnx_ir::ir::ElementType;
use proc_macro2::TokenStream;
use quote::quote;

/// Node for cast operations.
#[derive(Debug, Clone, new)]
pub struct CastNode {
    pub input: Type,
    pub output: Type,
    /// Target element type from ONNX cast operation
    pub target_elem_type: ElementType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for CastNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        use onnx_ir::ir::ElementType;

        match (&self.input, &self.output) {
            (Type::Scalar(input_scalar), Type::Scalar(output_scalar)) => {
                let input = &input_scalar.name;
                let output = &output_scalar.name;

                // Determine target scalar kind from the ONNX element type
                let target_kind = match self.target_elem_type {
                    ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                        ScalarKind::Float32
                    }
                    ElementType::Int32 | ElementType::Int64 => ScalarKind::Int64,
                    ElementType::Bool => ScalarKind::Bool,
                    ElementType::String => panic!("Cast: String type not supported"),
                };

                if input_scalar.kind == target_kind {
                    // If the input and target types are the same, we don't need to cast.
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Cast to the target type specified by ONNX
                    let ty = match self.target_elem_type {
                        ElementType::Float32 | ElementType::Float16 => quote! { f32 },
                        ElementType::Float64 => quote! { f64 },
                        ElementType::Int32 => quote! { i32 },
                        ElementType::Int64 => quote! { i64 },
                        ElementType::Bool => quote! { bool },
                        ElementType::String => panic!("Cast: String type not supported"),
                    };
                    quote! {
                        let #output = #input as #ty;
                    }
                }
            }
            (Type::Tensor(input_tensor), Type::Tensor(output_tensor)) => {
                let input = scope.tensor_use_owned(input_tensor, node_position);
                let output = &output_tensor.name;

                // Determine target tensor kind from the ONNX element type
                let target_kind = match self.target_elem_type {
                    ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                        TensorKind::Float
                    }
                    ElementType::Int32 | ElementType::Int64 => TensorKind::Int,
                    ElementType::Bool => TensorKind::Bool,
                    ElementType::String => panic!("Cast: String type not supported"),
                };

                if input_tensor.kind == target_kind {
                    // If the input and target types are the same, we don't need to cast.
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Cast to the target type specified by ONNX
                    let cast_fn = match target_kind {
                        TensorKind::Bool => quote! { bool() },
                        TensorKind::Int => quote! { int() },
                        TensorKind::Float => quote! { float() },
                    };
                    quote! {
                        let #output = #input.#cast_fn;
                    }
                }
            }
            (Type::Shape(input_shape), Type::Shape(output_shape)) => {
                let input = &input_shape.name;
                let output = &output_shape.name;
                // Shape types are always represented as [i64; N] in Burn
                // Even when casting to int32, we keep them as i64 arrays
                // The target_elem_type is ignored for Shape types
                quote! {
                    let #output = #input;
                }
            }
            (Type::Shape(input_shape), Type::Tensor(output_tensor)) => {
                // Cast Shape to Tensor (only for float types)
                let input = &input_shape.name;
                let output = &output_tensor.name;
                let rank = input_shape.rank;

                // Only convert to tensor for float types
                // For int types, this should have been handled in onnx-ir as Shape->Shape
                match self.target_elem_type {
                    ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let float_array: [f32; #rank] = shape_array.map(|x| x as f32);
                                Tensor::<B, 1>::from_data(
                                    TensorData::from(float_array),
                                    &self.device
                                )
                            };
                        }
                    }
                    ElementType::Bool => {
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let bool_array: [bool; #rank] = shape_array.map(|x| x != 0);
                                Tensor::<B, 1, Bool>::from_data(
                                    TensorData::from(bool_array),
                                    &self.device
                                )
                            };
                        }
                    }
                    ElementType::Int32 | ElementType::Int64 => {
                        // This shouldn't happen - onnx-ir should keep Shape as Shape for int casts
                        panic!(
                            "Cast: Shape to Int tensor should be handled as Shape->Shape in onnx-ir"
                        )
                    }
                    ElementType::String => panic!("Cast: String type not supported"),
                }
            }
            _ => panic!("Cast: unsupported type combination"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Cast(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_cast_scalar_f64_to_f32() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(CastNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float64)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float32)),
            onnx_ir::ir::ElementType::Float32,
        ));

        graph.register_input_output(vec!["scalar1".to_string()], vec!["scalar2".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, scalar1: f64) -> f32 {
                    let scalar2 = scalar1 as f32;
                    scalar2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_cast_tensor_float_to_int() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(CastNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_int("tensor2", 4)),
            onnx_ir::ir::ElementType::Int64,
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Int> {
                    let tensor2 = tensor1.int();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_cast_tensor_int_to_bool() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(CastNode::new(
            Type::Tensor(TensorType::new_int("tensor1", 2)),
            Type::Tensor(TensorType::new_bool("tensor2", 2)),
            onnx_ir::ir::ElementType::Bool,
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 2, Int>) -> Tensor<B, 2, Bool> {
                    let tensor2 = tensor1.bool();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
