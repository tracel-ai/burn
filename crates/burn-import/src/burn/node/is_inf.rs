use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::is_inf::IsInfConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct IsInfNode {
    pub input: Type,
    pub output: Type,
    pub config: IsInfConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for IsInfNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = match &self.input {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("Input must be a tensor or scalar"),
        };
        let output = &self.output.name();

        let function = match &self.output {
            Type::Scalar(_) => match (self.config.detect_negative, self.config.detect_positive) {
                (true, true) => quote! { #input.is_infinite() },
                (false, true) => quote! { #input.is_infinite() && #input.is_sign_positive() },
                (true, false) => quote! { #input.is_infinite() && #input.is_sign_negative() },
                (false, false) => quote! { false },
            },
            Type::Tensor(_) => match (self.config.detect_negative, self.config.detect_positive) {
                (true, true) => quote! { #input.is_inf() },
                (false, true) => {
                    quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                }
                (true, false) => {
                    quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                }
                (false, false) => quote! { #input.zeros_like().bool() },
            },
            _ => panic!("IsInf only supports scalar or tensor outputs"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::IsInf(self)
    }
}

impl OnnxIntoNode for IsInfNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs, config) = match node {
            onnx_ir::Node::IsInf {
                inputs,
                outputs,
                config,
                ..
            } => (inputs, outputs, config),
            _ => panic!("Expected IsInf node"),
        };
        let input = Type::from(inputs.first().unwrap());
        let output = Type::from(outputs.first().unwrap());
        Self::new(input, output, config.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };

    #[test]
    fn test_codegen_is_inf_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(IsInfNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_bool("tensor2", 4)),
            IsInfConfig::new(true, true),
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

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.is_inf();
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_is_inf_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(IsInfNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
            IsInfConfig::new(true, true),
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

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = scalar1.is_infinite();
                    scalar2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
