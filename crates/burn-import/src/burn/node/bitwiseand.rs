use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseAndNode {
    pub inputs: Vec<Type>,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseAndNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        self.inputs.clone()
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = match &self.inputs[0] {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("BitwiseAndNode only supports tensor and scalar inputs"),
        };
        
        let rhs = match &self.inputs[1] {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            _ => panic!("BitwiseAndNode only supports tensor and scalar inputs"),
        };

        let output = &self.output.name;

        // Check if the second input is a scalar
        let operation = match &self.inputs[1] {
            Type::Scalar(_) => quote! { #lhs.bitwise_and_scalar(#rhs.elem()) },
            Type::Tensor(_) => quote! { #lhs.bitwise_and(#rhs) },
            _ => panic!("BitwiseAndNode only supports tensor and scalar inputs"),
        };

        quote! {
            let #output = #operation;
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitwiseAndNode only supports Int TensorType outputs");
        }
        Node::BitwiseAnd(self)
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
        node::{bitwiseand::BitwiseAndNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitwise_and() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitwiseAndNode {
            inputs: vec![
                TensorType::new_int("input1", 1),
                TensorType::new_int("input2", 1),
            ],
            output: TensorType::new_int("output", 1),
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
                pub fn forward(&self, input1: Tensor<B, 1, Int>, input2: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
                    let output = input1 & input2;
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}
