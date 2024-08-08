use onnx_ir::ir::Data as OnnxData;
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
    pub val: Option<TensorData>,
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    //Float16(Vec<f16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Bool(Vec<bool>),
}
impl TensorData {
    fn len(&self) -> usize {
        match self {
            TensorData::Float32(data) => data.len(),
            TensorData::Float64(data) => data.len(),
            //TensorData::Float16(data) => data.len(),
            TensorData::Int32(data) => data.len(),
            TensorData::Int64(data) => data.len(),
            TensorData::Bool(data) => data.len(),
        }
    }

    fn as_tokens(&self) -> Vec<TokenStream> {
        match self {
            TensorData::Float32(data) => data.iter().map(|x| quote! { #x }).collect(),
            TensorData::Float64(data) => data.iter().map(|x| quote! { #x }).collect(),
            //TensorData::Float16(data) => data.iter().map(|x| quote! { #x }).collect(),
            TensorData::Int32(data) => data.iter().map(|x| quote! { #x }).collect(),
            TensorData::Int64(data) => data.iter().map(|x| quote! { #x }).collect(),
            TensorData::Bool(data) => data.iter().map(|x| quote! { #x }).collect(),
        }
    }
}

impl From<OnnxData> for TensorData {
    fn from(data: OnnxData) -> Self {
        match data {
            OnnxData::Float32s(data) => TensorData::Float32(data),
            OnnxData::Float64s(data) => TensorData::Float64(data),
            OnnxData::Int32s(data) => TensorData::Int32(data),
            OnnxData::Int64s(data) => TensorData::Int64(data),
            OnnxData::Bools(data) => TensorData::Bool(data),
            //OnnxData::Float16s(data) => TensorData::Float16(data),
            OnnxData::Float16(_) | OnnxData::Float16s(_) => {
                panic!("Float16 not supported for constant tensors")
            }
            _ => panic!("Expected Vector of numeric data, got {:?}", data),
        }
    }
}

impl Iterator for TensorData {
    type Item = TokenStream;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TensorData::Float32(data) => data.pop().map(|x| quote! { #x }),
            TensorData::Float64(data) => data.pop().map(|x| quote! { #x }),
            //TensorData::Float16(data) => data.pop().map(|x| quote! { #x }),
            TensorData::Int32(data) => data.pop().map(|x| quote! { #x }),
            TensorData::Int64(data) => data.pop().map(|x| quote! { #x }),
            TensorData::Bool(data) => data.pop().map(|x| quote! { #x }),
        }
    }
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
    pub dim: usize,
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
}

impl ScalarType {
    pub fn new<S: AsRef<str>>(name: S, kind: ScalarKind) -> Self {
        if name.as_ref().is_empty() {
            panic!("Scalar of Type {:?} was passed with empty name", kind);
        }
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

impl ShapeType {
    pub fn new<S: AsRef<str>>(name: S, dim: usize) -> Self {
        if name.as_ref().is_empty() {
            panic!("Shape was passed with empty name");
        }
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
        }
    }
    pub fn ty(&self) -> TokenStream {
        let dim = self.dim.to_tokens();
        quote! { [usize; #dim] }
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
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
            kind,
            shape,
            val: None,
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
            } => quote! { Tensor<B, #dim> },
            TensorType {
                kind: TensorKind::Int,
                ..
            } => quote! { Tensor<B, #dim, Int> },

            TensorType {
                kind: TensorKind::Bool,
                ..
            } => quote! { Tensor<B, #dim, Bool> },
        }
    }
    /// Note on the order of dims:
    /// The order of dims is reversed from the order of the shape.
    /// Consider two tensors with shapes [3, 1, 1] and [1, 1, 3].
    /// The resulting stream of tokens should be:
    /// [3, 1, 1] -> [[[v]], [[v]], [[v]]]
    /// [1, 1, 3] -> [[[v, v, v]]]
    pub fn val(&self) -> TokenStream {
        if let Some(val) = &self.val {
            let val = val.as_tokens();
            if let Some(shape) = &self.shape {
                // let's just handle the case where the shape is a single value
                if shape.len() == 1 {
                    if shape[0] != val.len() {
                        panic!(
                            "Tensor {:?} has shape {:?} but value has length {:?}",
                            self.name,
                            shape,
                            val.len()
                        );
                    }
                    return self.tensor_internal(self.render_row(&val).clone());
                }
                let take_n = shape.last().unwrap();
                let mut chunks = val.chunks_exact(*take_n);

                let mut result = Vec::new();
                //for each dimension, we need to iterate over all the following dimensions
                for i in (0..shape.len() - 1).rev() {
                    for j in (i..shape.len() - 1).rev() {
                        let dim = shape[j];
                        let mut tmp = Vec::new();
                        for _ in 0..dim {
                            tmp.push(self.render_row(chunks.next().unwrap()));
                        }
                        //treat the lower dimensions as a value in a row
                        result.push(self.render_row(&tmp));
                    }
                }
                return self.tensor_internal(self.render_row(&result));
            } else {
                panic!(
                    "Tensor {:?} has no shape, likely should be scalar or shape",
                    self.name
                );
            }
        } else {
            quote! {}
        }
    }
    fn render_row(&self, row: &[TokenStream]) -> TokenStream {
        quote! {
            [#(#row),*]
        }
    }
    fn tensor_internal(&self, tok: TokenStream) -> TokenStream {
        //what to do for bools?
        let dim = self.dim;
        return match self.kind {
            TensorKind::Int => quote! {
                Tensor::<B,#dim>::from_ints(#tok, &self.device)
            },
            TensorKind::Float => quote! {
                Tensor::<B,#dim>::from_floats(#tok,&self.device)
            },
            TensorKind::Bool => quote! {
                Tensor::<B,#dim>::from_data(#tok, &self.device)
            },
        };
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
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            ty: tokens,
        }
    }
    pub fn ty(&self) -> TokenStream {
        self.ty.clone()
    }
}
