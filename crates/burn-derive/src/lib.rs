#![warn(missing_docs)]

//! The derive crate of Burn.

#[macro_use]
extern crate derive_new;

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record_state;
pub(crate) mod shared;

/// Derive macro for the `Module` trait.
///
/// # Sub-modules
///
/// By default, the macro automatically detects sub-modules and parameters as module types.
///
/// Any field not recognized as a module type is assumed to be a non-module
/// and is skipped by the module system (not persistent, not visited).
///
/// ## Generics
///
/// Generic type parameters (e.g., `field: M`) are assumed to be sub-modules by default.
/// If a generic field represents some other runtime state or configuration, you can use
/// the `#[module(skip)]` attribute to provide a hint.
///
/// # Field Attributes
///
/// ## `#[module(skip)]`
///
/// Explicitly marks a field to be ignored by the module derive.
///
/// Skipped fields are not parameters, not modules, and are not persistent.
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
/// pub struct MyModule<M, N: NonModuleTrait> {
///     /// A normal parameter.
///     weights: Param<Tensor<2>>,
///     /// A field configured at runtime.
///     dropout_prob: f64,
///     /// A field that is recomputed at runtime.
///     cached_mask: Option<Tensor<2>>,
///     /// A field that contains some debug state.
///     debug_state: String,
///     /// Treated as a module (default for generics).
///     inner: M,
///     /// Hint required: this generic is NOT a module.
///     #[module(skip)]
///     other: N,
/// }
/// ```
#[proc_macro_derive(Module, attributes(module))]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module::derive_impl(&input)
}

/// Derive macro for the config.
#[proc_macro_derive(Config, attributes(config))]
pub fn config_derive(input: TokenStream) -> TokenStream {
    let item = syn::parse(input).unwrap();
    config::derive_impl(&item)
}

/// Derive macro for a recordable state (optimizer or learning-rate scheduler), decomposing it
/// into named tensors and scalars for the burnpack format.
///
/// Supported field shapes: `Tensor<D>`, `Option<Tensor<D>>`, `Vec<Tensor<D>>`, scalars
/// (`usize`/`isize`/`u8`..`u64`/`i8`..`i64`/`f32`/`f64`/`bool`), `Option<scalar>`, a nested
/// `RecordState`, and `Option<Nested>`. Scalar fields must use a concrete primitive type, not a
/// type alias (e.g. `f64`, not `LearningRate`): classification is syntactic, so an alias is treated
/// as a nested state.
#[proc_macro_derive(RecordState)]
pub fn record_state_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    record_state::derive_impl(&input)
}
