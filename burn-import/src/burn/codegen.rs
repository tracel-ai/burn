use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

use super::Scope;

#[derive(Default)]
pub struct FieldNameGenerator {
    conv2d_count: usize,
    matmul_count: usize,
}

#[derive(Default)]
pub struct BurnImports {
    tensor: bool,
    conv2d: bool,
}

impl BurnImports {
    pub fn register_conv2d(&mut self) {
        self.conv2d = true;
    }

    pub fn register_tensor(&mut self) {
        self.tensor = true;
    }
}

impl FieldNameGenerator {
    pub fn gen_conv2d(&mut self) -> Ident {
        self.conv2d_count += 1;
        let name = format!("conv2d_{}", self.conv2d_count);
        Ident::new(&name, Span::call_site())
    }

    pub fn gen_matmul(&mut self) -> Ident {
        self.matmul_count += 1;
        let name = format!("matmul_{}", self.matmul_count);
        Ident::new(&name, Span::call_site())
    }
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

pub trait NodeCodegen: std::fmt::Debug {
    fn output_type(&self) -> TokenStream;
    fn output_name(&self) -> Ident;
    fn input_def(&self) -> TokenStream;
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    fn field_name(&self) -> Option<Ident>;
    fn new_body(&self) -> TokenStream;
    fn new_field(&self) -> TokenStream;
    fn input_tensors(&self) -> Vec<Ident>;
    fn output_tensors(&self) -> Vec<Ident>;
}

impl<const N: usize, T: Copy + quote::ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        let mut body = quote! {};

        for i in 0..N {
            let elem = self[i];
            body.extend(quote! {#elem,});
        }

        quote! {
            [#body]
        }
    }
}

impl ToTokens for usize {
    fn to_tokens(&self) -> TokenStream {
        let value = self.to_string();
        let stream: proc_macro2::TokenStream = value.parse().unwrap();

        stream
    }
}
