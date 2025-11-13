use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::trilu::TriluConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TriluNode {
    pub input: TensorType,
    pub output: TensorType,
    pub config: TriluConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for TriluNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }
    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;
        let diagonal = self.config.diagonal.to_tokens();
        if self.config.upper {
            quote! {
                let #output = #input.triu(#diagonal);
            }
        } else {
            quote! {
                let #output = #input.tril(#diagonal);
            }
        }
    }
    fn into_node(self) -> super::Node<PS> {
        Node::Trilu(self)
    }
}

impl OnnxIntoNode for TriluNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let output = TensorType::from(node.outputs.first().unwrap());
        let config = node.config::<onnx_ir::node::trilu::TriluConfig>().clone();
        Self::new(input, output, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{test::assert_tokens, trilu::TriluConfig, trilu::TriluNode},
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_triu() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TriluConfig::new(true, 0);
        graph.register(TriluNode::new(
            TensorType::new_float("input", 2),
            TensorType::new_float("output", 2),
            config,
        ));
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
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.triu(0);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_tril() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TriluConfig::new(false, 0);
        graph.register(TriluNode::new(
            TensorType::new_float("input", 2),
            TensorType::new_float("output", 2),
            config,
        ));
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
                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let output = input.tril(0);
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}
