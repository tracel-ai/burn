use super::ConfigAnalyzerFactory;
use quote::quote;

pub(crate) fn derive_impl(item: &syn::DeriveInput) -> proc_macro::TokenStream {
    let factory = ConfigAnalyzerFactory::new();
    let analyzer = factory.create_analyzer(item);

    let constructor = analyzer.gen_new_fn();
    let module_name = analyzer.module_name();
    let builders = analyzer.gen_builder_fns();
    let serde = analyzer.gen_serde_impl();
    let clone = analyzer.gen_clone_impl();
    let display = analyzer.gen_display_impl();
    let config_impl = analyzer.gen_config_impl();

    quote! {
        mod #module_name {
            use super::*;
            use burn::serde as serde;

            #config_impl
            #constructor
            #builders
            #serde
            #clone
            #display
        }
    }
    .into()
}
