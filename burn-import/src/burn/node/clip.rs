use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct ClipNode {
    pub input: TensorType,
    pub output: TensorType,
    pub min: Option<f64>, // Should be elem Type
    pub max: Option<f64>,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ClipNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name;

        if let Some(min) = self.min {
            if let Some(max) = self.max {
                quote! {
                    let #output = #input.clamp(#min.elem(), #max.elem());
                }
            } else {
                quote! {
                    let #output = #input.clamp_min(#min.elem());
                }
            }
        } else if let Some(max) = self.max {
            return quote! {
                let #output = #input.clamp_max(#max.elem());
            };
        } else {
            panic!("Clip node must have at least one min or max value");
        }
    }
    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::ElementConversion");
    }

    fn into_node(self) -> Node<PS> {
        Node::Clip(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{graph::BurnGraph, node::test::assert_tokens, TensorType};

    #[test]
    fn codegen_nodes_min_max() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            Some(0.0),
            Some(1.0),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::ElementConversion;
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
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp(0f64.elem(), 1f64.elem());

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn codegen_nodes_min() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            Some(0.0),
            None,
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::ElementConversion;
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
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp_min(0f64.elem());

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn codegen_nodes_max() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ClipNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            None,
            Some(1.0),
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

        let expected = quote! {
            use burn::tensor::ElementConversion;
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
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                    }
                }
                #[allow(clippy::let_and_return)]
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.clamp_max(1f64.elem());

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
