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
//! - `dispatch` lowers Burn's built-in `Dispatch` implementations.
//! - `extension` lowers user-defined backend extension traits.
//! - `ir` describes tensor inputs, outputs, and backend calls independently of either frontend.
//! - `routing` owns shared backend selection, input extraction, invocation, and output wrapping.
//! - `derive` maps extension structs and enums across the dispatch boundary.
//! - `catalog` is the single list of runtime backends used by generated and handwritten paths.

use proc_macro::TokenStream;

mod catalog;
mod derive;
mod dispatch;
mod extension;
mod fusion;
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
/// Struct and enum inputs derive [`ExtensionType`] and use `#[extension_type]` on the corresponding
/// method argument. Autodiff support for custom operations requires a handwritten implementation of the extension trait
/// for Autodiff<B, C>; this macro does not generate backward passes.
///
/// # Fusion
///
/// Add `Fusion` (optionally `Fusion: cfg(...)`) to also generate a lazy implementation
/// for `Fusion<B>`. Enable Burn's `fusion` feature and choose a behavior for each method:
///
/// - `#[fusion(dtype = lhs, shape = lhs)]`: describe a single tensor output using field expressions.
/// - `#[fusion(meta = callable)]`: compute output metadata now and defer the inner backend call.
/// - `#[fusion(default)]`: inherit the trait's existing default body.
///
/// Choose exactly one form. Output metadata is computed before registration; execution is deferred.
///
/// ## Field expressions
///
/// Tensor names refer to `DType` values in `dtype` and `&Shape` values in `shape`.
/// Extension arguments refer to borrowed companion metadata; ordinary arguments are borrowed.
/// Both fields are required.
///
/// A bare operand copies its shape. Function calls such as `shape = output_shape(lhs, rhs)`
/// and inline blocks return an owned `Shape`. Use a block for an inline calculation;
/// standalone closures are not invoked automatically.
///
/// ```rust,ignore
/// #[backend_extension(Cube, Fusion)]
/// pub trait MyExtension: Backend {
///     #[fusion(dtype = input, shape = {
///         let mut shape = input.clone();
///         shape.swap(0, 1);
///         shape
///     })]
///     fn transpose_2d(input: FloatTensor<Self>) -> FloatTensor<Self>;
/// }
/// ```
///
/// ## Complete metadata
///
/// `meta` accepts a function path or closure receiving borrowed `burn::backend::fusion::custom::TensorSpec`
/// values, extension metadata, and ordinary arguments in declaration order. Its result mirrors the
/// outputs: specs for tensors, tuples for tuples, and companions from [`ExtensionType`] with
/// `#[extension_type(fusion)]` for structs and enums. Enum metadata selects the output variant.
/// For example, `#[fusion(meta = |x| (x.clone(), x.clone()))]` describes two tensors matching `x`,
/// while `#[fusion(meta = |cache| cache.clone())]` preserves a structured input's layout.
///
/// ## Optimizer integration
///
/// The operation ID defaults to the method name; use `id = "custom_matmul"` to override it.
/// Integer parameters (8–64 bits, `usize`, `isize`), `f32`, `f64`, and `bool` are exposed to custom
/// optimizers automatically, in declaration order. These are host values, not tensor contents.
/// Other ordinary arguments and extension fields are captured for execution only.
///
/// Mark aliases or types convertible to `burn::backend::Scalar` with `#[fusion(scalar)]`.
/// For custom encodings, annotate the parameter with `#[fusion(scalar = strategy.to_code())]`.
/// The expression returns a value convertible to `Scalar` and runs before inputs are consumed.
/// The original argument is still passed to the backend; marked and inferred scalars share
/// declaration order.
///
/// ## Requirements
///
/// Methods using field expressions or `meta` must have:
///
/// - A synchronous, non-generic signature.
/// - At least one input tensor, directly or inside an owned extension value. Primitive inputs may be borrowed.
/// - Owned ordinary arguments implementing `Clone + Send + Sync + 'static`.
/// - Output metadata computable without tensor readback.
///
/// Wrong dtype categories are rejected before registration; actual output shape, dtype, or device
/// mismatches become execution errors, as do output variants differing from their metadata.
/// Ordinary output fields come from metadata; the inner backend's values are discarded without
/// comparison. Custom kernels are opaque unless a custom optimizer recognizes their IR.
///
/// For other signatures, use an existing default body or omit `Fusion` from `#[backend_extension]`
/// and implement the trait for `Fusion<B>` manually.
#[proc_macro_attribute]
pub fn backend_extension(attr: TokenStream, item: TokenStream) -> TokenStream {
    extension::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Maps structs and enums of backend tensor primitives across the `Dispatch` boundary.
///
/// Opt into `#[extension_type(fusion)]` (optionally `fusion: cfg(...)`) to support Fusion inputs
/// and outputs and generate a backend-independent `NameMetadata` companion. It mirrors the struct's
/// fields or enum's variants, replacing tensors with `TensorSpec` and nested extension values with
/// their metadata. Ordinary fields are cloned and must implement `Clone + Debug`;
/// captured metadata must also be `Send + Sync + 'static`.
///
/// Ordinary output fields are taken from metadata. Their values returned by the inner backend are
/// discarded without comparison; the metadata callback must supply the intended public values.
/// Output variants and ordinary fields must be determined from metadata before execution.
/// Empty variants are supported, but each operation still needs an input tensor for its device.
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
