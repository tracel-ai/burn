use quote::quote;

pub fn display_fn(_ast: &syn::DeriveInput) -> proc_macro2::TokenStream {
    quote! {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(self, Default::default());
            write!(f, "{}", formatted)
        }
    }
}
