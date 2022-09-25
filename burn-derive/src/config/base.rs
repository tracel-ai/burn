use super::ConfigAnalyzerFactory;
use proc_macro2::TokenStream;
use quote::quote;

pub(crate) fn config_attr_impl(item: &syn::DeriveInput) -> TokenStream {
    let factory = ConfigAnalyzerFactory::new();
    let analyzer = factory.create_analyzer(item);

    let constructor = analyzer.gen_constructor_impl();
    let builders = analyzer.gen_builder_fn_impl();
    let serde = analyzer.gen_serde();
    let clone = analyzer.gen_clone();
    let display = analyzer.gen_display();
    let config_impl = analyzer.gen_config_impl();

    quote! {
        #config_impl
        #constructor
        #builders
        #serde
        #clone
        #display
    }
}
