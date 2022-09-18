use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod shared;

use config::config_attr_impl;
use module::module_derive_impl;

#[proc_macro_derive(Module)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module_derive_impl(&input)
}

#[proc_macro_derive(Config, attributes(config))]
pub fn config_derive(input: TokenStream) -> TokenStream {
    let item = syn::parse(input).unwrap();
    config_attr_impl(&item).into()
}
