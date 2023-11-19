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
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorKind {
    Int,
    Float,
    Bool,
}

#[derive(Debug, Clone)]
pub enum ScalarKind {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
}

#[derive(Debug, Clone)]
pub struct ScalarType {
    pub name: Ident,
    pub kind: ScalarKind,
}

#[derive(Debug, Clone)]
pub struct OtherType {
    pub name: Ident,
    pub ty: TokenStream,
}

#[derive(Debug, Clone)]
pub enum Type {
    /// Tensor type.
    Tensor(TensorType),

    /// Scalar type.
    Scalar(ScalarType),

    // Other type (more flexible type).
    Other(OtherType),
}

impl Type {
    pub fn name(&self) -> &Ident {
        match self {
            Type::Tensor(tensor) => &tensor.name,
            Type::Scalar(scalar) => &scalar.name,
            Type::Other(other) => &other.name,
        }
    }
    pub fn ty(&self) -> TokenStream {
        match self {
            Type::Tensor(tensor) => tensor.ty(),
            Type::Scalar(scalar) => scalar.ty(),
            Type::Other(other) => other.ty(),
        }
    }
}

impl ScalarType {
    pub fn new<S: AsRef<str>>(name: S, kind: ScalarKind) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            kind,
        }
    }
    pub fn ty(&self) -> TokenStream {
        match self.kind {
            ScalarKind::Int32 => quote! { i32 },
            ScalarKind::Int64 => quote! { i64 },
            ScalarKind::Float32 => quote! { f32 },
            ScalarKind::Float64 => quote! { f64 },
            ScalarKind::Bool => quote! { bool },
        }
    }
}

impl TensorType {
    pub fn new<S: AsRef<str>>(
        name: S,
        dim: usize,
        kind: TensorKind,
        shape: Option<Vec<usize>>,
    ) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
            kind,
            shape,
        }
    }
    pub fn new_float<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Float, None)
    }

    pub fn new_int<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Int, None)
    }

    pub fn new_bool<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new(name, dim, TensorKind::Bool, None)
    }

    pub fn ty(&self) -> TokenStream {
        let dim = self.dim.to_tokens();
        match self {
            TensorType {
                kind: TensorKind::Float,
                ..
            } => quote! {
                Tensor<B, #dim>
            },
            TensorType {
                kind: TensorKind::Int,
                ..
            } => quote! {
                Tensor<B, #dim, Int>
            },
            TensorType {
                kind: TensorKind::Bool,
                ..
            } => quote! {
                Tensor<B, #dim, Bool>
            },
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
