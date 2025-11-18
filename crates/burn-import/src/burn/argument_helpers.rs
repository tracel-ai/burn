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
            match tensor.dtype {
                DType::F32 | DType::F64 | DType::F16 => quote! { Tensor<B, #rank> },
                DType::I32 | DType::I64 | DType::I8 | DType::U16 | DType::U8 => {
                    quote! { Tensor<B, #rank, Int> }
                }
                DType::Bool => quote! { Tensor<B, #rank, Bool> },
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

/// Convert a shape argument to a tensor on the device
/// Uploads the Shape to the device as a rank 1 Int tensor
pub fn shape_to_tensor(arg: &Argument) -> TokenStream {
    let shape_name = Ident::new(&arg.name, Span::call_site());
    quote! { Tensor::<B, 1, burn::tensor::Int>::from_data(&#shape_name as &[_], &*self.device) }
}

/// Convert a scalar argument to a full tensor with the given shape
/// Uploads the Scalar to the device as a full tensor using the given shape definition
pub fn scalar_to_full_tensor(arg: &Argument, shape: &[usize]) -> TokenStream {
    if let ArgType::Scalar(dtype) = &arg.ty {
        let name = Ident::new(&arg.name, Span::call_site());
        let shape_tokens = shape
            .iter()
            .map(ToTokens::to_tokens)
            .map(|s| quote! {#s, })
            .collect::<TokenStream>();
        let rank = shape.len();
        let rank_tokens = rank.to_tokens();
        let tensor_kind = match dtype {
            DType::I32 | DType::I64 => quote! { burn::tensor::Int },
            DType::F32 | DType::F64 => quote! { burn::tensor::Float },
            DType::Bool => quote! { burn::tensor::Bool },
            _ => panic!("Unsupported scalar dtype for full tensor: {:?}", dtype),
        };
        quote! {
            Tensor::<B, #rank_tokens, #tensor_kind>::full([#shape_tokens], #name, &*self.device)
        }
    } else {
        panic!(
            "scalar_to_full_tensor called on non-scalar argument: {:?}",
            arg
        );
    }
}

/// Get the argument identifier
pub fn arg_ident(arg: &Argument) -> Ident {
    Ident::new(&arg.name, Span::call_site())
}
