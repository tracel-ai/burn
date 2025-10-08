use super::{Node, NodeCodegen, OnnxIntoNode};
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
            // -----------------------
            // Scalar -> Scalar
            // -----------------------
            (Type::Scalar(input_scalar), Type::Scalar(output_scalar)) => {
                let input = &input_scalar.name;
                let output = &output_scalar.name;

                // Map ONNX element types to Burn ScalarKind "families".
                // Burn scalar kinds are coarse (Float32/Float64/Int64/Bool), so:
                // - Float16 is lowered to Float32 "family"
                // - All integer widths map to Int64 "family" for equality checks
                let target_kind = match self.target_elem_type {
                    ElementType::Float64 => ScalarKind::Float64,
                    ElementType::Float32 | ElementType::Float16 => ScalarKind::Float32,
                    ElementType::Int32
                    | ElementType::Int64
                    | ElementType::Int8
                    | ElementType::Uint16
                    | ElementType::Uint8 => ScalarKind::Int64,
                    ElementType::Bool => ScalarKind::Bool,
                    ElementType::String => panic!("Cast: String type not supported"),
                };

                if input_scalar.kind == target_kind {
                    // No-op cast within same scalar "family".
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Concrete Rust target type for the cast expression.
                    let ty = match self.target_elem_type {
                        ElementType::Float32 | ElementType::Float16 => quote! { f32 },
                        ElementType::Float64 => quote! { f64 },
                        ElementType::Int32 => quote! { i32 },
                        ElementType::Int64 => quote! { i64 },
                        ElementType::Uint16 => quote! { u16 },
                        ElementType::Int8 => quote! { i8 },
                        ElementType::Uint8 => quote! { u8 },
                        ElementType::Bool => quote! { bool },
                        ElementType::String => panic!("Cast: String type not supported"),
                    };
                    quote! {
                        let #output = #input as #ty;
                    }
                }
            }

            // -----------------------
            // Tensor -> Tensor
            // -----------------------
            (Type::Tensor(input_tensor), Type::Tensor(output_tensor)) => {
                let input = scope.tensor_use_owned(input_tensor, node_position);
                let output = &output_tensor.name;

                // Map ONNX element types to Burn TensorKind categories.
                // Burn only distinguishes Float / Int / Bool at the Tensor level.
                let target_kind = match self.target_elem_type {
                    ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                        TensorKind::Float
                    }
                    ElementType::Int32
                    | ElementType::Int64
                    | ElementType::Int8
                    | ElementType::Uint16
                    | ElementType::Uint8 => TensorKind::Int,
                    ElementType::Bool => TensorKind::Bool,
                    ElementType::String => panic!("Cast: String type not supported"),
                };

                if input_tensor.kind == target_kind {
                    // No-op cast if already in the correct TensorKind category.
                    quote! {
                        let #output = #input;
                    }
                } else {
                    // Burn exposes category-level casts: .float(), .int(), .bool()
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

            // -----------------------
            // Shape -> Shape
            // -----------------------
            (Type::Shape(input_shape), Type::Shape(output_shape)) => {
                let input = &input_shape.name;
                let output = &output_shape.name;
                // Shapes stay as [i64; N] regardless of ONNX target. No cast.
                quote! {
                    let #output = #input;
                }
            }

            // -----------------------
            // Shape -> Tensor
            // (Mostly for float/bool visualization or downstream ops.)
            // -----------------------
            (Type::Shape(input_shape), Type::Tensor(output_tensor)) => {
                let input = &input_shape.name;
                let output = &output_tensor.name;
                let rank = input_shape.rank;

                match self.target_elem_type {
                    ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                        // Emit f32 tensor; Float64 target collapses to f32 at runtime side.
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
                    // For all integer widths (Int32, Int64, Int8, Uint8), we keep Shape as Shape
                    // in onnx-ir and shouldn't go through Shape->Tensor here.
                    ElementType::Int32
                    | ElementType::Int64
                    | ElementType::Int8
                    | ElementType::Uint16
                    | ElementType::Uint8 => {
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

impl OnnxIntoNode for CastNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = crate::burn::Type::from(node.inputs.first().unwrap());
        let output = crate::burn::Type::from(node.outputs.first().unwrap());
        let config = onnx_ir::node::cast::cast_config(&node);
        Self::new(input, output, config.to)
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
