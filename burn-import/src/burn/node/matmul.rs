use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorDescription, ToTokens};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;
use syn::Ident;

#[derive(Debug, Clone, new)]
pub struct MatmulNode {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub output: TensorDescription,
}

impl Serialize for MatmulNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_none()
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for MatmulNode {
    fn output_type(&self) -> TokenStream {
        let dim = self.output.dim.to_tokens();

        quote! {
            Tensor<B, #dim>
        }
    }

    fn output_name(&self) -> Ident {
        self.output.name.clone()
    }

    fn input_def(&self) -> TokenStream {
        let name_lhs = &self.lhs.name;
        let name_rhs = &self.rhs.name;
        let dim_lhs = self.lhs.dim.to_tokens();
        let dim_rhs = self.rhs.dim.to_tokens();

        quote! {
            #name_lhs: Tensor<B, #dim_lhs>, #name_rhs: Tensor<B, #dim_rhs>
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let lhs = scope.use_owned_tensor(&self.lhs.name, node_position);
        let rhs = scope.use_owned_tensor(&self.rhs.name, node_position);
        let output = &self.output.name;

        quote! {
            let #output = #lhs.matmul(#rhs);
        }
    }

    fn input_tensors(&self) -> Vec<Ident> {
        vec![self.lhs.name.clone(), self.rhs.name.clone()]
    }

    fn output_tensors(&self) -> Vec<Ident> {
        vec![self.output.name.clone()]
    }

    fn into_node(self) -> Node<PS> {
        Node::Matmul(self)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        graph::Graph,
        node::{matmul::MatmulNode, test::assert_tokens},
        TensorDescription,
    };

    #[test]
    fn test_codegen_two_nodes() {
        let mut graph = Graph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorDescription::new("tensor1", 4),
            TensorDescription::new("tensor2", 4),
            TensorDescription::new("tensor3", 4),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model <B: Backend>{}

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    Self { }
                }

                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.matmul(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
