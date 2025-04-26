use super::{Node, NodeCodegen};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Node for all unary operators.
#[derive(Debug, Clone)]
pub struct ConstantOfShapeNode {
    pub input: Type,
    pub output: Type,
    pub value: ConstantValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// Float constant.
    Float32(f32),
    Float64(f64),

    /// Integer constant.
    Int32(i32),
    Int64(i64),

    // Boolean constant.
    Bool(bool),
}

impl ConstantOfShapeNode {
    pub fn new(input: Type, output: Type, value: ConstantValue) -> Self {
        assert!(
            matches!(input, Type::Shape(_)),
            "ConstantOfShape input needs to be a Shape!"
        );
        assert!(
            matches!(output, Type::Tensor(_)),
            "ConstantOfShape output needs to be a Tensor!"
        );
        Self {
            input,
            output,
            value,
        }
    }
}

impl ConstantValue {
    pub fn val_tokens(&self) -> TokenStream {
        match self {
            Self::Float32(val) => quote! { #val },
            Self::Float64(val) => quote! { #val },
            Self::Int32(val) => quote! { #val },
            Self::Int64(val) => quote! { #val },
            Self::Bool(val) => quote! { #val },
        }
    }

    pub fn from_vec<T: Into<Self> + Copy>(mut source: Vec<T>) -> Self {
        assert_eq!(
            source.len(),
            1,
            "ConstantOfShape value from a vec needs to have exactly 1 element!"
        );
        source.drain(..).next().unwrap().into()
    }
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        Self::Float32(value)
    }
}
impl From<f64> for ConstantValue {
    fn from(value: f64) -> Self {
        Self::Float64(value)
    }
}
impl From<i32> for ConstantValue {
    fn from(value: i32) -> Self {
        Self::Int32(value)
    }
}
impl From<i64> for ConstantValue {
    fn from(value: i64) -> Self {
        Self::Int64(value)
    }
}
impl From<bool> for ConstantValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantOfShapeNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = self.output.name();
        let input = self.input.name();

        let output_rank = match &self.output {
            Type::Tensor(tensor) => tensor.rank.to_tokens(),
            _ => unreachable!(),
        };

        let value = self.value.val_tokens();
        // Note: in the generated code, self.device is a &module::Ignored<Device>,
        // so to get a &Device, &* is needed

        match &self.value {
            ConstantValue::Bool(bool) => {
                // Currently there is no full bool tensor support in the backend
                // So we use 0 or 1 with bool type casting
                // See: https://github.com/tracel-ai/burn/issues/1535
                if *bool {
                    quote! {
                        let #output = Tensor::<B, #output_rank, Int>::ones(#input, &*self.device).bool();
                    }
                } else {
                    quote! {
                        let #output = Tensor::<B, #output_rank, Int>::zeros(#input, &*self.device).bool();
                    }
                }
            }
            _ => quote! {
                let #output = Tensor::full(#input, #value, &*self.device);
            },
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::ConstantOfShape(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ShapeType, TensorType,
        graph::BurnGraph,
        node::{constant_of_shape::ConstantOfShapeNode, test::assert_tokens},
    };

    #[test]
    fn test_constant_val() {
        assert_eq!(ConstantValue::from(1i32), ConstantValue::Int32(1i32));
        assert_eq!(ConstantValue::from(-1i64), ConstantValue::Int64(-1i64));
        assert_eq!(ConstantValue::from(0f32), ConstantValue::Float32(0f32));
        assert_eq!(ConstantValue::from(0f64), ConstantValue::Float64(0f64));
        assert_eq!(ConstantValue::from(true), ConstantValue::Bool(true));
        assert_eq!(
            ConstantValue::from_vec(vec![2i32]),
            ConstantValue::Int32(2i32)
        );
    }

    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            Type::Shape(ShapeType::new("shape1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            ConstantValue::Float32(1.25f32),
        ));

        graph.register_input_output(vec!["shape1".to_string()], vec!["tensor2".to_string()]);

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
                pub fn forward(&self, shape1: [usize;4]) -> Tensor<B, 4> {
                    let tensor2 = Tensor::full(shape1, 1.25f32, &*self.device);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
