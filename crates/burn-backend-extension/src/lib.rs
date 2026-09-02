//! Procedural macros for Burn's runtime backend dispatch.
//!
//! The crate has two frontends backed by one code-generation pipeline:
//!
//! ```text
//! #[backend_dispatch]  ─┐
//!                       ├─> ir::Operation ─> routing ─> generated enum dispatch
//! #[backend_extension] ─┘                       │
//!                                               └─> ExtensionType mapping for structs/enums
//! ```
//!
//! - [`dispatch`] lowers Burn's built-in `Dispatch` implementations.
//! - [`extension`] lowers user-defined backend extension traits.
//! - [`ir`] describes tensor inputs, outputs, and backend calls independently of either frontend.
//! - [`routing`] owns shared backend selection, input extraction, invocation, and output wrapping.
//! - [`derive`] maps extension structs and enums across the dispatch boundary.
//! - [`catalog`] is the single list of runtime backends used by generated and handwritten paths.

use proc_macro::TokenStream;

mod catalog;
mod derive;
mod dispatch;
mod extension;
mod ir;
mod routing;

pub(crate) use catalog::{BACKENDS, BackendSpec};

/// Injects the backend catalog into a callback macro for handwritten dispatch paths.
///
/// This crate owns the authoritative backend list and derives its distributed subset and transfer
/// matrix. `burn-dispatch` provides only the local wrapper and callback macros that consume them.
#[doc(hidden)]
#[proc_macro]
pub fn backend_catalog(input: TokenStream) -> TokenStream {
    catalog::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Turns a backend-generic `impl Trait<Self> for Dispatch` into direct enum dispatch.
///
/// Routing binds `B` to the selected backend, then executes the forwarding body:
///
/// ```rust,ignore
/// #[backend_dispatch]
/// impl BoolTensorOps<Self> for Dispatch {
///     fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
///         B::bool_not(tensor)
///     }
/// }
/// ```
///
/// Methods requiring bespoke routing can use `#[backend_dispatch(skip)]`.
#[doc(hidden)]
#[proc_macro_attribute]
pub fn backend_dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    dispatch::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Generates the `Dispatch` implementation for a backend extension trait.
///
/// The backend comes from one routing tensor, preferring a float. The autodiff contexts of all
/// tensor-bearing inputs are merged; disabled inputs act as constants, while enabled inputs must
/// share a gradient-checkpointing strategy.
///
/// ```rust,ignore
/// #[backend_extension(Autodiff, Wgpu)]
/// pub trait MyExtension: Backend {
///     fn fused(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
/// }
/// ```
///
/// Struct and enum inputs derive [`ExtensionType`] and use `#[extension_type]` on the corresponding
/// method argument. Their autodiff implementation remains handwritten on `Autodiff<B, C>`.
#[proc_macro_attribute]
pub fn backend_extension(attr: TokenStream, item: TokenStream) -> TokenStream {
    extension::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Maps structs and enums of backend tensor primitives across the `Dispatch` boundary.
///
/// Tensor fields are mapped automatically. Nested extension values must be marked with
/// `#[extension_type]`; other fields pass through unchanged.
///
/// ```rust,ignore
/// #[derive(ExtensionType)]
/// pub struct Inputs<B: Backend> {
///     pub lhs: FloatTensor<B>,
///     pub rhs: FloatTensor<B>,
/// }
/// ```
#[proc_macro_derive(ExtensionType, attributes(extension_type))]
pub fn derive_extension_type(input: TokenStream) -> TokenStream {
    derive::expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
