use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;

use crate::burn::ToTokens;

#[derive(Debug, Clone)]
pub struct TensorType {
    pub name: Ident,
    pub dim: usize,
    pub kind: TensorKind,
}

#[derive(Debug, Clone)]
pub enum TensorKind {
    Int,
    Float,
    Bool,
}

#[derive(Debug, Clone)]
pub struct OtherType {
    pub name: Ident,
    pub ty: TokenStream,
}

pub enum Type<'a> {
    Tensor(&'a TensorType),
    Other(&'a OtherType),
}

impl<'a> Type<'a> {
    pub fn name(&self) -> &Ident {
        match self {
            Type::Tensor(tensor) => &tensor.name,
            Type::Other(other) => &other.name,
        }
    }
    pub fn ty(&self) -> TokenStream {
        match self {
            Type::Tensor(tensor) => tensor.ty(),
            Type::Other(other) => other.ty(),
        }
    }
}

impl TensorType {
    pub fn new<S: AsRef<str>>(name: S, dim: usize, kind: TensorKind) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
            kind,
        }
    }
    pub fn new_float<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Float)
    }

    pub fn new_int<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Int)
    }

    pub fn new_bool<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Bool)
    }

    pub fn ty(&self) -> TokenStream {
        let dim = self.dim.to_tokens();

        quote! {
            Tensor<B, #dim>
        }
    }
}

impl OtherType {
    pub fn new<S: AsRef<str>>(name: S, tokens: TokenStream) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            ty: tokens,
        }
    }
    pub fn ty(&self) -> TokenStream {
        self.ty.clone()
    }
}
