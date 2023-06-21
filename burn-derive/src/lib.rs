#![warn(missing_docs)]

//! The derive crate of Burn.

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record;
pub(crate) mod shared;

use config::config_attr_impl;
use module::module_derive_impl;
use record::record_derive_impl;

/// Derive macro for the module.
#[proc_macro_derive(Module)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module_derive_impl(&input)
}

/// Derive macro for the record.
#[proc_macro_derive(Record)]
pub fn record_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    record_derive_impl(&input)
}

/// Derive macro for the config.
#[proc_macro_derive(Config, attributes(config))]
pub fn config_derive(input: TokenStream) -> TokenStream {
    let item = syn::parse(input).unwrap();
    config_attr_impl(&item).into()
}
