use proc_macro2::TokenStream;
use quote::quote;

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

/// Padding configuration
impl ToTokens for PaddingConfig1d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig1d::Valid },
            Self::Explicit(padding) => {
                let padding = padding.to_tokens();
                quote! { PaddingConfig1d::Explicit(#padding) }
            }
        }
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig2d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig2d::Valid },
            Self::Explicit(padding1, padding2) => {
                let padding1 = padding1.to_tokens();
                let padding2 = padding2.to_tokens();
                quote! { PaddingConfig2d::Explicit(#padding1, #padding2) }
            }
        }
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig3d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig3d::Valid },
            Self::Explicit(padding1, padding2, padding3) => {
                let padding1 = padding1.to_tokens();
                let padding2 = padding2.to_tokens();
                let padding3 = padding3.to_tokens();
                quote! { PaddingConfig3d::Explicit(#padding1, #padding2, #padding3) }
            }
        }
    }
}
