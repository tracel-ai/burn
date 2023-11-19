use proc_macro2::Ident;
use quote::quote;

pub fn display_fn(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}[num_params={}]", stringify!(#name), self.num_params())

        }
    }
}
