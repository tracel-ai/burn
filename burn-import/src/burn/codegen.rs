use proc_macro2::TokenStream;
use quote::quote;

use burn::nn::PaddingConfig2d;

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

impl<const N: usize, T: Copy + ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        let mut body = quote! {};

        self.iter().for_each(|item| {
            let elem = item.to_tokens();
            body.extend(quote! {#elem,});
        });

        quote! {
            [#body]
        }
    }
}

/// Prettier output
impl ToTokens for usize {
    fn to_tokens(&self) -> TokenStream {
        let value = self.to_string();
        let stream: proc_macro2::TokenStream = value.parse().unwrap();

        stream
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig2d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Same => quote! { PaddingConfig2d::Same },
            Self::Valid => quote! { PaddingConfig2d::Valid },
            Self::Explicit(padding1, padding2) => {
                let padding1 = padding1.to_tokens();
                let padding2 = padding2.to_tokens();
                quote! { PaddingConfig2d::Explicit(#padding1, #padding2) }
            }
        }
    }
}
