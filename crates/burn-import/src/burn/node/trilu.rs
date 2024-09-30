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
        // #(#var)* — no separators
        // #(#var),* — the character before the asterisk is used as a separator
        // #( struct #var; )* — the repetition can contain other tokens
        // #( #k => println!("{}", #v), )* — even multiple interpolations
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
        node::{trilu::TriluConfig, trilu::TriluNode, test::assert_tokens},
        TensorType,
    };
    #[test]
    fn test_codegen_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(
            TriluNode::new(TensorType::new_int("input", 2), TensorType::new_int("output", 2), TriluConfig { upper: true, diagonal: 1 })
        );        
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

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, tensor1: Tensor<B, 2>) -> Tensor<B, 2> {
                    let tensor2 = tensor1.trilu(1);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}