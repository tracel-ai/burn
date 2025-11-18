use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorKind, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct BitwiseNotNode {
    pub input: TensorType,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BitwiseNotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![{
            if self.input.kind != TensorKind::Int {
                panic!("BitwiseNotNode only supports Int TensorType inputs");
            }
            Type::Tensor(self.input.clone())
        }]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #input.bitwise_not();
        }
    }

    fn into_node(self) -> Node<PS> {
        if self.output.kind != TensorKind::Int {
            panic!("BitwiseNotNode only supports Int TensorType outputs");
        }
        Node::BitwiseNot(self)
    }
}

impl OnnxIntoNode for BitwiseNotNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::BitwiseNot(n) = node else {
            panic!("Expected BitwiseNot node");
        };
        let input = crate::burn::TensorType::from(n.inputs.first().unwrap());
        let output = crate::burn::TensorType::from(n.outputs.first().unwrap());
        Self::new(input, output)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{bitwisenot::BitwiseNotNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_bitwise_not() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(BitwiseNotNode {
            input: TensorType::new_int("input", 2),
            output: TensorType::new_int("output", 2),
        });
        graph.register_input_output(
            vec!["input".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
                    let output = input.bitwise_not();
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}
