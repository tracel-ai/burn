use proc_macro::TokenStream;
use quote::{format_ident, quote};

#[allow(missing_docs)]
#[proc_macro_attribute]
pub fn testgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: proc_macro2::TokenStream = proc_macro2::TokenStream::from(item);
    let attr: proc_macro2::TokenStream = proc_macro2::TokenStream::from(attr);
    let macro_ident = format_ident!("testgen_{}", attr.to_string());

    let macro_gen = quote! {
        #[macro_export]
        macro_rules! #macro_ident {
            () => {
                mod #attr {
                    use super::*;

                    #item
                }
            };
        }
    };

    macro_gen.into()
}
