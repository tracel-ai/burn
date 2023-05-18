use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;

#[derive(Debug, Clone)]
pub struct TensorType {
    pub name: Ident,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct OtherType {
    pub name: Ident,
    pub ty: TokenStream,
}

pub enum Type {
    Tensor(TensorType),
    Other(OtherType),
}

impl TensorType {
    pub fn new<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
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
}
