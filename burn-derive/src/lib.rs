use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod shared;

use config::config_attr_impl;
use module::module_derive_impl;

#[proc_macro_derive(Module)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    module_derive_impl(&ast)
}

#[proc_macro_derive(Config, attributes(config))]
pub fn config_derive(item: TokenStream) -> TokenStream {
    let item = syn::parse(item).unwrap();

    // panic!("{}", tokens);
    config_attr_impl(&item).into()
}
