use proc_macro2::TokenStream;
use quote::quote;

use onnx_ir::ir::DType;
use onnx_ir::node::padding::{PaddingConfig1d, PaddingConfig2d, PaddingConfig3d};

// ============================================================================
// Codegen utilities for converting types to TokenStream
// ============================================================================

fn convert_primitive<T: core::fmt::Debug>(primitive: T) -> TokenStream {
    let value = format!("{primitive:?}");

    value.parse().unwrap()
}

fn convert_to_array<'a, I, T>(list: I) -> TokenStream
where
    I: Iterator<Item = &'a T>,
    T: ToTokens + 'a,
{
    let mut body = quote! {};

    list.for_each(|item| {
        let elem = item.to_tokens();
        body.extend(quote! {#elem,});
    });

    quote! {
        [#body]
    }
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

impl<const N: usize, T: Copy + ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

impl<T: Copy + ToTokens> ToTokens for Vec<T> {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

/// Prettier output for `usize`
impl ToTokens for usize {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `i64`
impl ToTokens for i64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f64`
impl ToTokens for f64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f32`
impl ToTokens for f32 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Padding configuration for 1D operations.
///
/// Converts PaddingConfig1d to Rust code tokens.
/// Format: Explicit(left, right)
impl ToTokens for PaddingConfig1d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig1d::Valid },
            Self::Explicit(left, right) => {
                let left = left.to_tokens();
                let right = right.to_tokens();
                quote! { PaddingConfig1d::Explicit(#left, #right) }
            }
        }
    }
}

/// Converts PaddingConfig2d to Rust code tokens.
/// Format: Explicit(top, left, bottom, right)
impl ToTokens for PaddingConfig2d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig2d::Valid },
            Self::Explicit(top, left, bottom, right) => {
                let top = top.to_tokens();
                let left = left.to_tokens();
                let bottom = bottom.to_tokens();
                let right = right.to_tokens();
                quote! { PaddingConfig2d::Explicit(#top, #left, #bottom, #right) }
            }
        }
    }
}

/// Converts PaddingConfig3d to Rust code tokens.
/// Format: Explicit(front, top, left, back, bottom, right)
impl ToTokens for PaddingConfig3d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig3d::Valid },
            Self::Explicit(front, top, left, back, bottom, right) => {
                let front = front.to_tokens();
                let top = top.to_tokens();
                let left = left.to_tokens();
                let back = back.to_tokens();
                let bottom = bottom.to_tokens();
                let right = right.to_tokens();
                quote! { PaddingConfig3d::Explicit(#front, #top, #left, #back, #bottom, #right) }
            }
        }
    }
}

/// DType for specifying tensor element types in generated code.
///
/// Note: Flex32 and QFloat are intentionally not supported as they are Burn-specific
/// runtime types that cannot come from ONNX models. Flex32 is a GPU optimization type
/// and QFloat requires quantization schemes not representable in ONNX.
impl ToTokens for DType {
    fn to_tokens(&self) -> TokenStream {
        match self {
            DType::F16 => quote! { burn::tensor::DType::F16 },
            DType::BF16 => quote! { burn::tensor::DType::BF16 },
            DType::F32 => quote! { burn::tensor::DType::F32 },
            DType::F64 => quote! { burn::tensor::DType::F64 },
            DType::I8 => quote! { burn::tensor::DType::I8 },
            DType::I16 => quote! { burn::tensor::DType::I16 },
            DType::I32 => quote! { burn::tensor::DType::I32 },
            DType::I64 => quote! { burn::tensor::DType::I64 },
            DType::U8 => quote! { burn::tensor::DType::U8 },
            DType::U16 => quote! { burn::tensor::DType::U16 },
            DType::U32 => quote! { burn::tensor::DType::U32 },
            DType::U64 => quote! { burn::tensor::DType::U64 },
            DType::Bool => quote! { burn::tensor::DType::Bool },
            // Flex32 and QFloat are Burn-specific runtime types not present in ONNX models
            _ => panic!(
                "Unsupported dtype for ONNX code generation: {:?}. \
                 Flex32 and QFloat are Burn-specific runtime types.",
                self
            ),
        }
    }
}
