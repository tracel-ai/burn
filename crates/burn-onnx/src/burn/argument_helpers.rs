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

/// Generate function parameters from a slice of arguments
///
/// Produces: `name1: Type1, name2: Type2, ...`
pub fn codegen_fn_params(args: &[Argument]) -> TokenStream {
    let params: Vec<_> = args
        .iter()
        .map(|arg| {
            let name = arg_ident(arg);
            let ty = arg_type_tokens(arg);
            quote! { #name: #ty }
        })
        .collect();

    quote! { #(#params),* }
}

/// Generate return type from output arguments
///
/// Single output: `Type`
/// Multiple outputs: `(Type1, Type2, ...)`
pub fn codegen_return_type(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        arg_type_tokens(&outputs[0])
    } else {
        let types: Vec<_> = outputs.iter().map(arg_type_tokens).collect();
        quote! { (#(#types),*) }
    }
}

/// Generate return expression from output arguments
///
/// Single output: `name`
/// Multiple outputs: `(name1, name2, ...)`
pub fn codegen_return_expr(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        let name = arg_ident(&outputs[0]);
        quote! { #name }
    } else {
        let names: Vec<_> = outputs.iter().map(arg_ident).collect();
        quote! { (#(#names),*) }
    }
}
