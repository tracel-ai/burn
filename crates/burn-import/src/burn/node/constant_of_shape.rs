use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ConstantOfShapeNode {
    pub value: ConstantOfShapeValue,
    pub input: Type,
    pub output: Type,
}

#[derive(Debug, Clone, new)]
pub enum ConstantOfShapeValue {
    /// Float constant.
    Float32(f32),
    Float64(f64),

    /// Integer constant.
    Int32(i32),
    Int64(i64),

    // Boolean constant.
    Bool(bool),
}

impl ToTokens for ConstantOfShapeValue {
    fn to_tokens(&self) -> TokenStream {
        match self {
            ConstantOfShapeValue::Bool(val) => val.to_tokens(),
            ConstantOfShapeValue::Float32(val) => val.to_tokens(),
            ConstantOfShapeValue::Float64(val) => val.to_tokens(),
            ConstantOfShapeValue::Int32(val) => val.to_tokens(),
            ConstantOfShapeValue::Int64(val) => val.to_tokens(),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantOfShapeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name();
        let value = self.value.to_tokens();

        match (&self.input, &self.output) {
            (Type::Tensor(input), Type::Tensor(_)) => {
                let input = scope.tensor_use_owned(&input, node_position);
                quote! {
                    let #output = Tensor::full(#input.to_data().value, #value, &#input.device());
                }
            }
            (Type::Scalar(_), Type::Scalar(_)) => {
                quote! {
                    let #output = #value;
                }
            }
            _ => panic!(
                "Invalid input/output type ({:?}, {:?})",
                self.input, self.output
            ),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Int");
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
        graph::BurnGraph,
        node::{constant_of_shape::ConstantOfShapeNode, test::assert_tokens},
        ScalarType, TensorType,
    };

    #[test]
    fn test_codegen() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeValue::new_float32(1.25),
            Type::Tensor(TensorType::new_int("tensor1", 1)),
            Type::Tensor(TensorType::new_float("tensor2", 3)),
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
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 1, Int>) -> Tensor<B, 3> {
                    let tensor2 = Tensor::full(tensor1.to_data().value, 1.25, &tensor1.device());

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeValue::new_float64(1.25),
            Type::Scalar(ScalarType::new("scalar1", crate::burn::ScalarKind::Int64)),
            Type::Scalar(ScalarType::new("scalar2", crate::burn::ScalarKind::Float64)),
        ));

        graph.register_input_output(vec!["scalar1".to_string()], vec!["scalar2".to_string()]);

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, scalar1: i64) -> f64 {
                    let scalar2 = 1.25;

                    scalar2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
