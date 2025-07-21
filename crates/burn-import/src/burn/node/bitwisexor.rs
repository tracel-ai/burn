use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseXorNode {
    pub inputs: Vec<Type>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseXorNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs.clone()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let output = &self.output.name;

        let operation = match (&self.inputs[0], &self.inputs[1]) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = scope.tensor_use_owned(rhs_tensor, node_position);
                quote! { #lhs.bitwise_xor(#rhs) }
            }
            (Type::Tensor(lhs_tensor), Type::Scalar(rhs_scalar)) => {
                let lhs = scope.tensor_use_owned(lhs_tensor, node_position);
                let rhs = &rhs_scalar.name;
                quote! { #lhs.bitwise_xor_scalar(#rhs.elem()) }
            }
            (Type::Scalar(_lhs_scalar), Type::Tensor(_rhs_tensor)) => {
                panic!(
                    "BitwiseXorNode does not support scalar as first input and tensor as second input"
                )
            }
            (Type::Scalar(_), Type::Scalar(_)) => {
                panic!("BitwiseXorNode does not support both inputs as scalars")
            }
            _ => panic!("BitwiseXorNode only supports tensor and scalar inputs"),
        };

        quote! {
            let #output = #operation;
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitwiseXorNode only supports Int TensorType outputs");
        }
        Node::BitwiseXor(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register ElementConversion for scalar operations
        for input in &self.inputs {
            if matches!(input, Type::Scalar(_)) {
                imports.register("burn::tensor::ElementConversion");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitwisexor::BitwiseXorNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitwise_xor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitwiseXorNode {
            inputs: vec![
                Type::Tensor(TensorType::new_int("input1", 2)),
                Type::Tensor(TensorType::new_int("input2", 2)),
            ],
            output: TensorType::new_int("output", 2),
        });
        graph.register_input_output(
            vec!["input1".to_string(), "input2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                pub fn forward(
                    &self,
                    input1: Tensor<B, 2, Int>,
                    input2: Tensor<B, 2, Int>
                ) -> Tensor<B, 2, Int> {
                    let output = input1.bitwise_xor(input2);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
