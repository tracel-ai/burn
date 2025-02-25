use crate::burn::ToTokens;
use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct TensorType {
    pub name: Ident,
    pub rank: usize,
    pub kind: TensorKind,
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Int,
    Float,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarKind {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarType {
    pub name: Ident,
    pub kind: ScalarKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeType {
    pub name: Ident,
    pub rank: usize,
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

    /// Shape type.
    Shape(ShapeType),

    // Other type (more flexible type).
    Other(OtherType),
}

impl Type {
    // This is used, because types might have number literal name, which cannot be
    // used as a variable name.
    pub fn format_name(name: &str) -> String {
        let name_is_number = name.bytes().all(|digit| digit.is_ascii_digit());
        if name_is_number {
            format!("_{}", name)
        } else {
            name.to_string()
        }
    }
    pub fn name(&self) -> &Ident {
        match self {
            Type::Tensor(tensor) => &tensor.name,
            Type::Scalar(scalar) => &scalar.name,
            Type::Shape(shape) => &shape.name,
            Type::Other(other) => &other.name,
        }
    }
    pub fn ty(&self) -> TokenStream {
        match self {
            Type::Tensor(tensor) => tensor.ty(),
            Type::Scalar(scalar) => scalar.ty(),
            Type::Shape(shape) => shape.ty(),
            Type::Other(other) => other.ty(),
        }
    }
    pub fn as_tensor(&self) -> &TensorType {
        if let Self::Tensor(t) = self {
            t
        } else {
            panic!("Called Type::as_tensor on {self:?}!");
        }
    }
    pub fn as_scalar(&self) -> &ScalarType {
        if let Self::Scalar(s) = self {
            s
        } else {
            panic!("Called Type::as_scalar on {self:?}!");
        }
    }
    pub fn as_shape(&self) -> &ShapeType {
        if let Self::Shape(s) = self {
            s
        } else {
            panic!("Called Type::as_shape on {self:?}!");
        }
    }
}

impl ScalarType {
    pub fn new<S: AsRef<str>>(name: S, kind: ScalarKind) -> Self {
        if name.as_ref().is_empty() {
            panic!("Scalar of Type {:?} was passed with empty name", kind);
        }

        let formatted_name = Type::format_name(name.as_ref());
        Self {
            name: Ident::new(&formatted_name, Span::call_site()),
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

    /// Helper for Ops that need to process a Scalar as a tensor on device
    ///
    /// Uploads the Scalar to the device as a full tensor using the given shape definition
    pub fn to_full_tensor(&self, shape: &[usize]) -> TokenStream {
        let name = &self.name;
        let shape_tokens = shape
            .iter()
            .map(ToTokens::to_tokens)
            .map(|s| quote! {#s, })
            .collect::<TokenStream>();
        let rank = shape.len();
        let rank_tokens = rank.to_tokens();
        let tensor_kind = match self.kind {
            ScalarKind::Int32 | ScalarKind::Int64 => quote! { burn::tensor::Int },
            ScalarKind::Float32 | ScalarKind::Float64 => quote! { burn::tensor::Float },
            ScalarKind::Bool => quote! { burn::tensor::Bool },
        };
        quote! {
            Tensor::<B, #rank_tokens, #tensor_kind>::full([#shape_tokens], #name, &*self.device)
        }
    }
}

impl ShapeType {
    pub fn new<S: AsRef<str>>(name: S, dim: usize) -> Self {
        if name.as_ref().is_empty() {
            panic!("Shape was passed with empty name");
        }
        let formatted_name = Type::format_name(name.as_ref());
        Self {
            name: Ident::new(&formatted_name, Span::call_site()),
            rank: dim,
        }
    }
    pub fn ty(&self) -> TokenStream {
        let dim = self.rank.to_tokens();
        quote! { [usize; #dim] }
    }

    /// Helper for Ops that need to process a shape as a tensor on device
    ///
    /// Uploads the Shape to the device as a rank 1 Int tensor
    pub fn to_tensor(&self) -> TokenStream {
        let shape_name = &self.name;
        // To copy just the values from the shape value without moving it
        // (which could lead to ownership problems if the same Shape is used multiple times)
        // borrow the array as a slice and use that to create the Tensor:
        quote! { Tensor::<B, 1, burn::tensor::Int>::from_data(&#shape_name as &[_], &*self.device) }
    }
}

impl TensorType {
    pub fn new<S: AsRef<str>>(
        name: S,
        dim: usize,
        kind: TensorKind,
        shape: Option<Vec<usize>>,
    ) -> Self {
        if name.as_ref().is_empty() {
            panic!(
                "Tensor of Kind {:?} with dim shape {:?} was passed with empty name",
                kind, shape
            );
        }
        let formatted_name = Type::format_name(name.as_ref());

        assert_ne!(
            dim, 0,
            "Trying to create TensorType with dim = 0 - should be a Scalar instead!"
        );
        Self {
            name: Ident::new(&formatted_name, Span::call_site()),
            rank: dim,
            kind,
            shape,
        }
    }
    pub fn new_float<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new_float_with_shape(name, dim, None)
    }

    pub fn new_float_with_shape<S: AsRef<str>>(
        name: S,
        dim: usize,
        shape: Option<Vec<usize>>,
    ) -> Self {
        Self::new(name, dim, TensorKind::Float, shape)
    }

    pub fn new_int<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new_int_with_shape(name, dim, None)
    }

    pub fn new_int_with_shape<S: AsRef<str>>(
        name: S,
        dim: usize,
        shape: Option<Vec<usize>>,
    ) -> Self {
        Self::new(name, dim, TensorKind::Int, shape)
    }

    pub fn new_bool<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self::new_bool_with_shape(name, dim, None)
    }

    pub fn new_bool_with_shape<S: AsRef<str>>(
        name: S,
        dim: usize,
        shape: Option<Vec<usize>>,
    ) -> Self {
        Self::new(name, dim, TensorKind::Bool, shape)
    }

    pub fn ty(&self) -> TokenStream {
        let dim = self.rank.to_tokens();
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
        if name.as_ref().is_empty() {
            panic!(
                "Other type with tokens {:?} was passed with empty name",
                tokens
            );
        }
        let formatted_name = Type::format_name(name.as_ref());
        Self {
            name: Ident::new(&formatted_name, Span::call_site()),
            ty: tokens,
        }
    }
    pub fn ty(&self) -> TokenStream {
        self.ty.clone()
    }
}
