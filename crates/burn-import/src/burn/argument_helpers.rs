//! Helper functions for working with onnx_ir::Argument types
//!
//! This module provides utilities to generate code for different argument types
//! without needing the Type abstraction layer.

use onnx_ir::{
    Argument,
    ir::{ArgType, DType},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::burn::ToTokens;

/// Get the type TokenStream for an argument
pub fn arg_type_tokens(arg: &Argument) -> TokenStream {
    match &arg.ty {
        ArgType::Tensor(tensor) => {
            let rank = tensor.rank.to_tokens();
            match &tensor.dtype {
                dtype if dtype.is_float() => quote! { Tensor<B, #rank> },
                dtype if dtype.is_int() || dtype.is_uint() => {
                    quote! { Tensor<B, #rank, Int> }
                }
                dtype if dtype.is_bool() => quote! { Tensor<B, #rank, Bool> },
                _ => quote! { Tensor<B, #rank> },
            }
        }
        ArgType::Scalar(dtype) => scalar_type_tokens(dtype),
        ArgType::Shape(rank) => {
            let rank_lit = rank.to_tokens();
            quote! { [i64; #rank_lit] }
        }
    }
}

/// Get the type TokenStream for a scalar DType
pub fn scalar_type_tokens(dtype: &DType) -> TokenStream {
    match dtype {
        DType::I32 => quote! { i32 },
        DType::I64 => quote! { i64 },
        DType::F32 => quote! { f32 },
        DType::F64 => quote! { f64 },
        DType::Bool => quote! { bool },
        _ => panic!("Unsupported scalar dtype: {:?}", dtype),
    }
}

/// Get the argument identifier
pub fn arg_ident(arg: &Argument) -> Ident {
    Ident::new(&arg.name, Span::call_site())
}
