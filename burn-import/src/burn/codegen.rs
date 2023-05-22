use proc_macro2::TokenStream;
use quote::quote;

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
