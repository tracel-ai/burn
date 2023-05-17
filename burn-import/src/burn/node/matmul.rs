use super::Node;
use crate::burn::{Scope, TensorDescription, ToTokens};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

#[derive(Debug, Clone, new)]
pub struct Matmul {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub output: TensorDescription,
}

impl Node for Matmul {
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
}
