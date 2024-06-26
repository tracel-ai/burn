#![warn(missing_docs)]

//! The derive crate of Burn.

#[macro_use]
extern crate derive_new;

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record;
pub(crate) mod shared;

/// Derive macro for the module.
#[proc_macro_derive(Module, attributes(module))]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module::derive_impl(&input)
}

/// Derive macro for the record.
#[proc_macro_derive(Record)]
pub fn record_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    record::derive_impl(&input)
}

/// Derive macro for the config.
#[proc_macro_derive(Config, attributes(config))]
pub fn config_derive(input: TokenStream) -> TokenStream {
    let item = syn::parse(input).unwrap();
    config::derive_impl(&item)
}
