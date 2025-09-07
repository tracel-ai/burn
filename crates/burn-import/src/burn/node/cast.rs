use super::{Node, NodeCodegen};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use onnx_ir::ir::ElementType;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct CastNode {
    pub input: Type,
    pub output: Type,
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
        let out_ident = self.output.name();

        match (&self.input, &self.output) {
            // ===== Tensor → Tensor =====
            (Type::Tensor(in_tt), Type::Tensor(_)) => {
                let input_owned = scope.tensor_use_owned(in_tt, node_position);

                let expr = match self.target_elem_type {
                    // Floats
                    ElementType::Float16 | ElementType::Float32 | ElementType::Float64 => {
                        quote! { (#input_owned).float() }
                    }
                    // Ints
                    ElementType::Int8
                    | ElementType::Uint8
                    | ElementType::Int32
                    | ElementType::Int64 => {
                        quote! { (#input_owned).int() }
                    }
                    // Bool (nonzero → true); Burn already provides .bool()
                    ElementType::Bool => {
                        quote! { (#input_owned).bool() }
                    }
                    ElementType::String => quote! { panic!("Cast: String not supported") },
                };

                quote! { let #out_ident = #expr; }
            }

            // ===== Scalar → Scalar =====
            (Type::Scalar(_), Type::Scalar(_)) => {
                let in_ident = self.input.name();

                let expr = match self.target_elem_type {
                    ElementType::Float16 | ElementType::Float32 => quote! { (#in_ident as f32) },
                    ElementType::Float64 => quote! { (#in_ident as f64) },
                    ElementType::Int8 | ElementType::Uint8 | ElementType::Int32 => {
                        quote! { (#in_ident as i32) }
                    }
                    ElementType::Int64 => quote! { (#in_ident as i64) },
                    ElementType::Bool => quote! { ((#in_ident) != 0) },
                    ElementType::String => quote! { panic!("Cast: String not supported") },
                };

                quote! { let #out_ident = #expr; }
            }

            // Anything else should've been normalized earlier.
            _ => quote! { panic!("Cast codegen: unsupported input/output kind pairing") },
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
