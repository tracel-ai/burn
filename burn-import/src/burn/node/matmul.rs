use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::burn::{NodeCodegen, TensorInput, TensorOutput, TensorReferences, ToTokens};

#[derive(Debug, Clone)]
pub struct Matmul {
    pub lhs: TensorInput,
    pub rhs: TensorInput,
    pub output: TensorOutput,
}

impl NodeCodegen for Matmul {
    fn output_type(&self) -> TokenStream {
        let dim = self.output.dim;

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
        let dim_lhs = self.lhs.dim;
        let dim_rhs = self.rhs.dim;

        quote! {
            #name_lhs: Tensor<B, #dim_lhs>, #name_rhs: Tensor<B, #dim_rhs>
        }
    }

    fn field_name(&self) -> Option<Ident> {
        None
    }

    fn new_body(&self) -> TokenStream {
        // No field
        quote! {}
    }

    fn new_field(&self) -> TokenStream {
        // No field
        quote! {}
    }

    fn forward(&self) -> TokenStream {
        let lhs = self.lhs.to_tokens();
        let rhs = self.rhs.to_tokens();
        let output = &self.output.name;

        quote! {
            let #output = #lhs.matmul(#rhs);
        }
    }
}

impl TensorReferences for Matmul {
    fn increate_input_ref_count(&mut self, names: &mut HashMap<Ident, usize>) {
        self.lhs.ref_count(names);
        self.rhs.ref_count(names);
    }

    fn decreate_output_ref_count(&mut self, names: &mut HashMap<Ident, usize>) {
        self.output.ref_count(names);
    }
}
