#![warn(missing_docs)]

//! Derive and procedural macros for Burn.
//!
//! Use these through `burn`, which re-exports them; this crate is not meant to be a direct
//! dependency.
//!
//! - `#[derive(Module)]` implements `Module` for a struct or enum of modules, parameters and
//!   constants. `#[module(skip)]` excludes a field.
//! - `#[derive(Config)]` makes a struct a serializable configuration with a generated `new`
//!   constructor and `with_*` setters for optional fields and fields marked `#[config(default = ...)]`.
//! - `#[derive(RecordState)]` decomposes an optimizer or scheduler state into named tensors and
//!   scalars for the burnpack format.
//!
//! The function-like macros below implement `burn::tensor::assert_shape!`,
//! `debug_assert_shape!` and `einsum!`, and are not part of the public API.

#[macro_use]
extern crate derive_new;

use proc_macro::TokenStream;

pub(crate) mod config;
pub(crate) mod einsum;
pub(crate) mod module;
pub(crate) mod pytree;
pub(crate) mod record_state;
pub(crate) mod shape;
pub(crate) mod shared;

/// Derive macro for the `Module` trait.
///
/// # Sub-modules
///
/// Fields are treated as sub-modules unless their type is recognized as a constant or they
/// are marked `#[module(skip)]`. Participating fields must implement `Module` and `ModuleDisplay`.
///
/// Constants are detected syntactically, including `usize`, `f32`, `bool`, `String` and
/// combinations such as `Option<usize>` and `Vec<f32>`. Unlike 0.21, unknown concrete types
/// are not automatically skipped. Use `#[module(skip)]` for custom config/state fields that
/// report missing `Module` / `ModuleDisplay` implementations. Do not skip trainable children.
///
/// ## Generics
///
/// Bounds are inferred on each field's type: `Wrapper<T>` must implement `Module` and
/// `ModuleDisplay`, but `T` need not implement them. Existing bounds, including custom
/// forward traits, are preserved. Use `#[module(skip)]` for generic config/state fields;
/// the same type parameter can also appear in a participating field.
///
/// Named and tuple structs support recursive children such as `Vec<Self>` and
/// `Option<Box<Self>>`. Enums support variants containing a single unnamed field.
///
/// # Field Attributes
///
/// ## `#[module(skip)]`
///
/// Explicitly marks a field to be ignored by the module derive.
///
/// Skipped fields are excluded from parameter traversal, checkpointing, device movement and
/// training/validation transitions. They are still cloned and included in debug display.
/// This is equivalent to the deprecated `Ignored<T>` wrapper.
///
/// ### Requirements
///
/// The field must implement: `Debug + Clone + Send`.
/// A `Param` field cannot be skipped.
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
///     #[module(skip)]
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

/// Derive macro for `burn_func::PyTree`.
///
/// The struct must be generic over a backend type parameter named `B` or bounded by `Backend`.
/// Non-differentiable fields can be marked `#[pytree(skip)]`; they are cloned unchanged when the
/// backend is rebound and must not depend on the backend type parameter.
#[proc_macro_derive(PyTree, attributes(pytree))]
pub fn pytree_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    pytree::derive_impl(&input)
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

/// Implementation of `burn_tensor::assert_shape!`. Not part of the public API.
#[doc(hidden)]
#[proc_macro]
pub fn __assert_shape(input: TokenStream) -> TokenStream {
    shape_macro(input, shape::Mode::Assert)
}

/// Implementation of `burn_tensor::debug_assert_shape!`. Not part of the public API.
#[doc(hidden)]
#[proc_macro]
pub fn __debug_assert_shape(input: TokenStream) -> TokenStream {
    shape_macro(input, shape::Mode::DebugAssert)
}

/// The input is `$crate, tensor, [pattern]`, see the wrappers in burn-tensor.
fn shape_macro(input: TokenStream, mode: shape::Mode) -> TokenStream {
    let input = syn::parse_macro_input!(input as shape::ShapeInput);
    shape::expand(input, mode).into()
}

/// Implementation of `burn_tensor::einsum!`. Not part of the public API.
#[doc(hidden)]
#[proc_macro]
pub fn __einsum(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as einsum::EinsumInput);
    einsum::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
