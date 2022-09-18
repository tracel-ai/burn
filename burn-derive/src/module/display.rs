use proc_macro2::Ident;
use quote::quote;

pub fn display_fn() -> proc_macro2::TokenStream {
    quote! {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}[num_params={}]", self.name(), self.num_params())
        }
    }
}

pub fn name_fn(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        fn name(&self) -> &str {
            stringify!(#name)
        }
    }
}
