use proc_macro2::Ident;
use quote::quote;

pub fn display_fn(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}[num_params={}]", stringify!(#name), self.num_params())

        }
    }
}
