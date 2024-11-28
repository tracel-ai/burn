use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Config, Debug)]
pub struct SplitConfig {
    pub axis: usize,
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
        self.outputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let axis = self.config.axis.to_tokens();

        let output_names = self
            .outputs
            .iter()
            .map(|output| scope.tensor_use_owned(output, node_position))
            .collect::<Vec<_>>();

        let split_code = if let Some(split_sizes) = &self.config.split_sizes {
            let split_sizes_tokens = split_sizes.to_tokens();
            quote! {
                let split_tensors = #input.split_with_sizes(#split_sizes_tokens, #axis);
            }
        } else {
            let num_outputs = self.outputs.len();
            let num_outputs_tokens = num_outputs.to_tokens();
            quote! {
                let tensor_size = #input.shape().dims[#axis];
                let split_size = (tensor_size + #num_outputs_tokens - 1) / #num_outputs_tokens;
                let split_tensors = #input.split(split_size, #axis);
            }
        };

        let assignments: Vec<TokenStream> = output_names
            .iter()
            .enumerate()
            .map(|(i, output_name)| {
                let idx = syn::Index::from(i);
                quote! {
                    let #output_name = split_tensors[#idx].clone();
                }
            })
            .collect();

        quote! {
            #split_code
            #(#assignments)*
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Split(self)
    }
}
