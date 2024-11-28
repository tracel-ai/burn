use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Config, Debug)]
pub struct SplitConfig {
    pub axis: usize,
    pub num_outputs: Option<usize>,
    pub split_sizes: Option<Vec<usize>>,
}

#[derive(Debug, Clone, new)]
pub struct SplitNode {
    pub input: TensorType,
    pub outputs: Vec<TensorType>,
    pub config: SplitConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SplitNode {
    fn output_types(&self) -> Vec<Type> {
        let tensor = &self.outputs[0];
        let dims = tensor.dim;

        let vec_tensor_type = quote! {vec<Tensor<B, #dims>>};

        let other_type = OtherType {
            name: syn::Ident::new("split_tensors", proc_macro2::Span::call_site()),
            ty: vec_tensor_type,
        };

        vec![Type::Other(other_type)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let axis = self.config.axis.to_tokens();

        let split_code = if let Some(split_sizes) = &self.config.split_sizes {
            let split_sizes_tokens = split_sizes.to_tokens();
            quote! {
                let split_tensors = #input.split_with_sizes(#split_sizes_tokens, #axis);
            }
        } else {
            let num_outputs = self.config.num_outputs.unwrap();
            let num_outputs_tokens = num_outputs.to_tokens();
            quote! {
                let split_tensors = #input.split(#num_outputs_tokens, #axis);
            }
        };

        quote! {
            #split_code
            split_tensors
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Split(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{split::SplitNode, test::assert_tokens},
        TensorType,
    };

    #[test]
    fn test_codegen_split() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(SplitNode::new(
            TensorType::new_float("tensor1", 2),
            vec![
                TensorType::new_float("tensor2", 2),
                TensorType::new_float("tensor3", 2),
            ],
            SplitConfig {
                axis: 0,
                num_outputs: Some(2),
                split_sizes: None,
            },
        ));

        graph.register_input_output(
            vec!["tensor1".to_string()],
            vec!["split_tensors".to_string()],
        );

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 2>,
                ) -> Vec<Tensor<B, 2> {
                    let split_tensors = burn::tensor::Tensor::split(tensor1, 2, 0);

                    split_tensors
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
