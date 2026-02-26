#![warn(missing_docs)]

//! The derive crate of Burn.

#[macro_use]
extern crate derive_new;

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record;
pub(crate) mod shared;

// TODO: change nn modules `#[module(skip)]` (currently used for backward compat)
// to `#[module(constant)` for persistent non-parameter fields.

/// Derive macro for the `Module` trait.
///
/// # Field Attributes
///
/// By default, every field in a `Module` is treated as a sub-module or a parameter.
/// You can change this behavior using the `#[module]` attribute.
///
/// ## `#[module(constant)]`
///
/// Marks the field as a constant value. Constants are not parameters or modules, but are persistent.
///
/// Use this for configuration or other persistent metadata that should be saved and loaded with the module.
///
/// ### Requirements
///
/// The field must implement: `Debug + Clone + Send + Serialize + DeserializeOwned`.
///
/// ## `#[module(skip)]`
///
/// Marks the field to be skipped by the module system. Skipped fields are
/// not parameters, not modules, and are not persistent.
///
/// Use this for fields that should not be saved or restored with the module,
/// such as markers, caches or other auxiliary runtime state.
///
/// This is equivalent to the deprecated `Ignored<T>` wrapper.
///
/// ### Requirements
///
/// The field must implement: `Debug + Clone + Send`.
///
/// # Example
///
/// ```ignore
/// #[derive(Module, Debug)]
/// pub struct MyModule<B: Backend> {
///     /// A normal parameter.
///     weights: Param<Tensor<B, 2>>,
///     /// A persistent config value.
///     #[module(constant)]
///     dropout_prob: f64,
///     /// A field that is recomputed at runtime.
///     #[module(skip)]
///     cached_mask: Option<Tensor<B, 2>>,
///     /// A field that contains some debug state.
///     #[module(skip)]
///     debug_state: String,
/// }
/// ```
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
