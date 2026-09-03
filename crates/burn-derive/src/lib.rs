#![warn(missing_docs)]

//! The derive crate of Burn.

#[macro_use]
extern crate derive_new;

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod record_state;
pub(crate) mod shape;
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

/// Binds tensor axis sizes to names while checking the rest of the shape.
///
/// `unpack_shape!(tensor, [B, T, 80])` evaluates to a tuple with one `usize` per bare
/// identifier, in pattern order. The other slots are checks: `=expr` compares the axis with an
/// in-scope `usize` expression, an integer literal compares with that value, and `_` skips
/// the axis.
///
/// The pattern length must equal the tensor rank, otherwise the call does not compile. Axis
/// checks are always-on runtime assertions; a mismatch panics with the axis, the expected and
/// actual sizes, and the full dims.
///
/// ```ignore
/// let (b, t) = unpack_shape!(x, [B, T, 80]);
/// let (t,) = unpack_shape!(mask, [=b, T]);
/// let (c,) = unpack_shape!(y, [_, _, C]);
/// ```
///
/// Inspired by the `burn-contracts` crate by Crutcher Dunnavant.
#[proc_macro]
pub fn unpack_shape(input: TokenStream) -> TokenStream {
    let call = input.to_string();
    let input = syn::parse_macro_input!(input as shape::ShapeInput);
    shape::expand(input, shape::Mode::Unpack, &call).into()
}

/// Asserts a tensor's rank at compile time and its axis sizes at runtime.
///
/// `assert_shape!(tensor, [=b, =t, 256])` accepts `=expr` slots (an in-scope `usize`
/// expression), integer literals, and `_` to skip an axis. Bare identifiers are rejected: use
/// [`unpack_shape!`] to bind names.
///
/// The pattern length must equal the tensor rank, otherwise the call does not compile. Axis
/// checks stay enabled in release builds; see [`debug_assert_shape!`] for the debug-only
/// variant.
///
/// ```ignore
/// assert_shape!(y, [=b, =t, =self.d_model]);
/// assert_shape!(mask, [=b, _]);
/// ```
///
/// Inspired by the `burn-contracts` crate by Crutcher Dunnavant.
#[proc_macro]
pub fn assert_shape(input: TokenStream) -> TokenStream {
    let call = input.to_string();
    let input = syn::parse_macro_input!(input as shape::ShapeInput);
    shape::expand(input, shape::Mode::Assert, &call).into()
}

/// Same as [`assert_shape!`], but the axis checks compile out unless `debug_assertions` is on.
///
/// The rank check is a type check and stays in every build. Like `debug_assert!`, the tensor
/// expression is not evaluated in release builds.
///
/// ```ignore
/// debug_assert_shape!(hidden, [=b, =t, =self.d_model]);
/// ```
#[proc_macro]
pub fn debug_assert_shape(input: TokenStream) -> TokenStream {
    let call = input.to_string();
    let input = syn::parse_macro_input!(input as shape::ShapeInput);
    shape::expand(input, shape::Mode::DebugAssert, &call).into()
}
