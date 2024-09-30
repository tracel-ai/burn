use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;


#[derive(Config, Debug)]
pub struct TriluConfig {
    pub upper: bool,
    pub diagonal: i64,
}

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


#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;
    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{test::assert_tokens, trilu::TriluConfig, trilu::TriluNode},
        TensorType,
    };

    #[test]
    fn test_codegen_triu() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TriluConfig::new(true, 0);  // Upper triangular, diagonal offset 0
        graph.register(TriluNode::new(
            TensorType::new_float("input", 2),   // Example input tensor type
            TensorType::new_float("output", 2),  // Example output tensor type
            config,
        ));
        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

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
                    let output = input.triu(0);  // Example of upper triangular
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_tril() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = TriluConfig::new(false, 0);  // Lower triangular, diagonal offset 0
        graph.register(TriluNode::new(
            TensorType::new_float("input", 2),   // Example input tensor type
            TensorType::new_float("output", 2),  // Example output tensor type
            config,
        ));
        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

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
                    let output = input.tril(0);  // Example of upper triangular
                    output
                }
            }
        };
        assert_tokens(graph.codegen(), expected);
    }
}