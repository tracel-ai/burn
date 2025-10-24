use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;

use crate::burn::ToTokens;

#[derive(Debug, Clone)]
pub struct TensorType {
    pub name: Ident,
    pub rank: usize,
    pub kind: TensorKind,
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

/// Represents a shape type in the ONNX model.
///
/// Shapes are represented as arrays of i64 values ([i64; N]) for several important reasons:
/// 1. **Negative indexing support**: ONNX operations like Slice, Gather, etc. support negative
///    indices to count from the end of dimensions (e.g., -1 means last element)
/// 2. **Large dimension support**: i64 provides sufficient range for very large tensor dimensions
/// 3. **ONNX spec compliance**: The ONNX specification uses int64 for shape-related operations
/// 4. **Unified type system**: Using a single type [i64; N] simplifies code generation and
///    eliminates the need for type conversions in shape arithmetic operations
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
    pub fn name(&self) -> &Ident {
        match self {
            Type::Tensor(tensor) => &tensor.name,
            Type::Scalar(scalar) => &scalar.name,
            Type::Shape(shape) => &shape.name,
            Type::Other(other) => &other.name,
        }
    }

    /// Clone this type with a new name (already sanitized from onnx-ir)
    pub fn with_name(&self, new_name: &str) -> Self {
        let ident = Ident::new(new_name, Span::call_site());

        match self {
            Type::Tensor(tensor) => Type::Tensor(TensorType {
                name: ident,
                rank: tensor.rank,
                kind: tensor.kind,
            }),
            Type::Scalar(scalar) => Type::Scalar(ScalarType {
                name: ident,
                kind: scalar.kind.clone(),
            }),
            Type::Shape(shape) => Type::Shape(ShapeType {
                name: ident,
                rank: shape.rank,
            }),
            Type::Other(other) => Type::Other(OtherType {
                name: ident,
                ty: other.ty.clone(),
            }),
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
            panic!("Scalar of Type {kind:?} was passed with empty name");
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
    pub fn new<S: AsRef<str>>(name: S, rank: usize) -> Self {
        if name.as_ref().is_empty() {
            panic!("Shape was passed with empty name");
        }
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            rank,
        }
    }

    pub fn ty(&self) -> TokenStream {
        let rank = self.rank.to_tokens();
        // Shape arrays use i64 as the element type to support:
        // - Negative indices in operations (e.g., -1 for last element)
        // - Large tensor dimensions without overflow
        // - Direct compatibility with ONNX spec which uses int64
        quote! { [i64; #rank] }
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
    pub fn new<S: AsRef<str>>(name: S, rank: usize, kind: TensorKind) -> Self {
        if name.as_ref().is_empty() {
            panic!("Tensor of Kind {kind:?} was passed with empty name");
        }
        assert_ne!(
            rank, 0,
            "Trying to create TensorType with dim = 0 - should be a Scalar instead!"
        );
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            rank,
            kind,
        }
    }
    pub fn new_float<S: AsRef<str>>(name: S, rank: usize) -> Self {
        Self::new(name, rank, TensorKind::Float)
    }

    pub fn new_int<S: AsRef<str>>(name: S, rank: usize) -> Self {
        Self::new(name, rank, TensorKind::Int)
    }

    pub fn new_bool<S: AsRef<str>>(name: S, rank: usize) -> Self {
        Self::new(name, rank, TensorKind::Bool)
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
            panic!("Other type with tokens {tokens:?} was passed with empty name");
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

// ============================================================================
// ONNX type conversions
// ============================================================================

impl From<&onnx_ir::ir::Argument> for TensorType {
    fn from(arg: &onnx_ir::ir::Argument) -> Self {
        use onnx_ir::ir::{ArgType, TensorType as OnnxTensorType};

        match &arg.ty {
            ArgType::Tensor(OnnxTensorType { dtype, rank, .. }) => {
                tensor_type_from_elem_and_rank(arg.name.clone(), dtype, *rank)
            }
            ArgType::Scalar(dtype) => {
                // Represent scalar as rank-0 tensor type of the appropriate kind
                tensor_type_from_elem_and_rank(arg.name.clone(), dtype, 0)
            }
            ArgType::Shape(_) => panic!("Cannot convert Shape to Burn TensorType"),
        }
    }
}

impl From<&onnx_ir::ir::Argument> for Type {
    fn from(arg: &onnx_ir::ir::Argument) -> Self {
        use onnx_ir::ir::ArgType;

        match &arg.ty {
            ArgType::Tensor(tensor) => {
                // Treat tensor with rank 0 as scalar
                if tensor.rank == 0 {
                    Type::Scalar(ScalarType::new(
                        arg.name.clone(),
                        ScalarKind::from(&tensor.dtype),
                    ))
                } else {
                    let kind: TensorKind = tensor.dtype.into();
                    let rank = tensor.rank;
                    let name = arg.name.clone();
                    Type::Tensor(TensorType::new(name, rank, kind))
                }
            }

            ArgType::Scalar(elem_type) => {
                Type::Scalar(ScalarType::new(arg.name.clone(), elem_type.into()))
            }
            ArgType::Shape(rank) => Type::Shape(ShapeType::new(arg.name.clone(), *rank)),
        }
    }
}

impl From<&onnx_ir::ir::DType> for ScalarKind {
    fn from(dtype: &onnx_ir::ir::DType) -> Self {
        use onnx_ir::ir::DType;

        match dtype {
            DType::F32 => ScalarKind::Float32,
            DType::F64 => ScalarKind::Float64,
            DType::I32 => ScalarKind::Int32,
            DType::I64 => ScalarKind::Int64,
            DType::Bool => ScalarKind::Bool,
            DType::U16 => ScalarKind::Int32,
            DType::I8 | DType::U8 => ScalarKind::Int32,
            DType::F16 => panic!("Float16 tensor unsupported"),
            _ => panic!("Unsupported DType for ScalarKind conversion: {:?}", dtype),
        }
    }
}

impl From<onnx_ir::ir::DType> for TensorKind {
    fn from(dtype: onnx_ir::ir::DType) -> Self {
        use onnx_ir::ir::DType;

        match dtype {
            DType::F32 => TensorKind::Float,
            DType::F64 => TensorKind::Float,
            DType::I32 => TensorKind::Int,
            DType::I64 => TensorKind::Int,
            DType::I8 | DType::U8 => TensorKind::Int,
            DType::Bool => TensorKind::Bool,
            _ => panic!("Unsupported tensor type"),
        }
    }
}

fn tensor_type_from_elem_and_rank(
    name: String,
    elem: &onnx_ir::ir::DType,
    rank: usize,
) -> TensorType {
    use onnx_ir::ir::DType;

    match elem {
        DType::U8 | DType::I8 | DType::U16 | DType::I32 | DType::I64 => {
            TensorType::new(name, rank, TensorKind::Int)
        }

        DType::F16 | DType::F32 | DType::F64 => {
            // If you have TensorType::new_float, use that; otherwise:
            // TensorType::new(name, rank, TensorKind::Float)
            TensorType::new(name, rank, TensorKind::Float)
        }

        DType::Bool => TensorType::new(name, rank, TensorKind::Bool),

        _ => panic!("Unsupported element type for Burn TensorType"),
    }
}
