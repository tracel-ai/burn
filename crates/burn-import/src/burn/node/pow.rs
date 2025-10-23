use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Type of power operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerType {
    /// Integer power (powi)
    Int,
    /// Float power (powf)
    Float,
}

/// Power node that handles both integer and float exponents
#[derive(Debug, Clone)]
pub struct PowNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
    pub power_type: PowerType,
}

impl PowNode {
    pub fn new(lhs: Type, rhs: Type, output: Type, power_type: PowerType) -> Self {
        Self {
            lhs,
            rhs,
            output,
            power_type,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for PowNode {
    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            _ => panic!("lhs must be a tensor"),
        };

        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor or scalar"),
        };

        let output = &self.output.name();

        let function = match (self.power_type, &self.rhs) {
            (PowerType::Int, Type::Tensor(_)) => quote! { #lhs.powi(#rhs) },
            (PowerType::Int, Type::Scalar(_)) => quote! { #lhs.powi_scalar(#rhs) },
            (PowerType::Float, Type::Tensor(_)) => quote! { #lhs.powf(#rhs) },
            (PowerType::Float, Type::Scalar(_)) => quote! { #lhs.powf_scalar(#rhs) },
            _ => panic!("Invalid power type combination"),
        };

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Pow(self)
    }
}

impl OnnxIntoNode for PowNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let lhs = Type::from(node.inputs.first().unwrap());
        let rhs = Type::from(node.inputs.get(1).unwrap());
        let output = Type::from(node.outputs.first().unwrap());

        // Determine power type based on RHS type
        let power_type = match &rhs {
            Type::Tensor(x) => match x.kind {
                TensorKind::Int => PowerType::Int,
                TensorKind::Float => PowerType::Float,
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            Type::Scalar(x) => match x.kind {
                ScalarKind::Int32 | ScalarKind::Int64 => PowerType::Int,
                ScalarKind::Float32 | ScalarKind::Float64 => PowerType::Float,
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            _ => panic!("pow function only supports RHS scalar or tensor types"),
        };

        Self::new(lhs, rhs, output, power_type)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{TensorType, graph::BurnGraph, node::test::assert_tokens};

    #[test]
    fn test_codegen_powi() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(PowNode::new(
            Type::Tensor(TensorType::new_int("tensor1", 4)),
            Type::Tensor(TensorType::new_int("tensor2", 4)),
            Type::Tensor(TensorType::new_int("tensor3", 4)),
            PowerType::Int,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 4, Int>, tensor2: Tensor<B, 4, Int>) -> Tensor<B, 4, Int> {
                    let tensor3 = tensor1.powi(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_powf() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(PowNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            Type::Tensor(TensorType::new_float("tensor3", 4)),
            PowerType::Float,
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.powf(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
