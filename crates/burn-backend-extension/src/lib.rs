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
//! - `parse` reads an extension's lazily executed methods; `fusion` emits their lazy Fusion
//!   operations from it, and `remote` the remote backend's client and server handlers.
//! - `catalog` is the single list of runtime backends used by generated and handwritten paths.

use proc_macro::TokenStream;

mod catalog;
mod derive;
mod dispatch;
mod extension;
mod fusion;
mod ir;
mod parse;
mod remote;
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
/// share a gradient-checkpointing strategy. Only an autodiff-associated float selects the autodiff
/// implementation; enabled integer and boolean inputs propagate context without requiring it.
///
/// Struct and enum inputs derive [`ExtensionType`] and use `#[extension_type]` on the corresponding
/// method argument. Autodiff support for custom operations requires a handwritten implementation of the extension trait
/// for Autodiff<B, C>; this macro does not generate backward passes.
///
/// # Backend selection
///
/// Execution backend selectors include `Cube`, `Flex`, and `Remote`.
/// `Cube` covers the CubeCL runtimes, including WGPU, CUDA, ROCm, and CPU; `Wgpu` and `Cuda`
/// are not selectors. Gate a selector with, for example, `Cube: cfg(feature = "wgpu")`;
/// this condition refers to the consuming crate's features. Enable Burn's `extension` feature
/// and the backend features you target.
///
/// Implement the extension trait on each selected backend. Add `Autodiff` to generate routing
/// to your `Autodiff<B, C>` implementation. A default trait body composing differentiable
/// operations can supply that implementation; the macro does not derive a custom backward pass.
/// Calling an extension on an unlisted runtime backend panics.
///
/// Expose a high-level `Tensor<D>` wrapper by calling the generated `Dispatch` implementation
/// with `Tensor::into_dispatch` and wrapping its output with `Tensor::from_dispatch`.
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
/// Extension arguments refer to borrowed extension metadata; ordinary arguments are borrowed.
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
/// outputs: specs for tensors, tuples for tuples, and metadata types generated by [`ExtensionType`] with
/// `#[extension_type(fusion)]` for structs and enums. Enum metadata selects the output variant.
/// For example, `#[fusion(meta = |x| (x.clone(), x.clone()))]` describes two tensors matching `x`,
/// while `#[fusion(meta = |cache| cache.clone())]` preserves a structured input's layout.
/// Tuple elements must themselves be tensors, derived extension values, or tuples of those types.
/// Plain scalar returns and tuples such as `(FloatTensor<Self>, u32)` are unsupported; put ordinary
/// output fields in a struct or enum deriving [`ExtensionType`] with Fusion enabled.
///
/// For struct and enum outputs, the callback supplies the **actual return values** for non-tensor
/// fields. Fusion returns these values without waiting for execution and discards the backend's
/// values for those fields without comparison. They must equal what a direct backend call returns.
/// If metadata supplies `count = 7` but the backend returns `count = 8`, enabling Fusion changes the
/// result. That is an incorrect extension implementation, and the generated code does not detect it.
/// Tensor shapes, dtypes, and enum variants must also agree with direct backend execution.
///
/// For example, a derived output struct with a tensor and a `count: usize` field can return the
/// tensor and its element count:
///
/// ```rust,ignore
/// #[fusion(meta = |input| CountedMetadata {
///     tensor: input.clone(),
///     count: input.shape.num_elements(),
/// })]
/// fn counted(input: FloatTensor<Self>) -> Counted<Self>;
/// ```
///
/// The backend must return the same count.
/// A count of nonzero elements depends on tensor contents and cannot be computed from `TensorSpec`.
/// Return content-dependent values as tensors, or write a Fusion implementation that waits for
/// computation to finish before returning them.
///
/// ## Optimizer integration
///
/// The operation ID defaults to the trait and method names, as in `"AttentionBackend::attend"`;
/// use `id = "custom_matmul"` to override it.
/// Integer parameters (8–64 bits, `usize`, `isize`), `f32`, `f64`, and `bool` are exposed to custom
/// optimizers automatically, in declaration order. These are host values, not tensor contents.
/// Other ordinary arguments and extension fields are captured for execution only.
///
/// Mark aliases or types convertible to `burn::backend::Scalar` with `#[fusion(scalar)]`.
/// For custom encodings, annotate the parameter with `#[fusion(scalar = strategy.to_code())]`.
/// The expression returns a value convertible to `Scalar` and runs before inputs are consumed.
/// The original argument is still passed to the backend; marked and inferred scalars share
/// declaration order. Add `decode = callable` beside the encoding to rebuild the argument from
/// its scalar, which the remote backend requires (see [Remote](#remote)).
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
/// Debug builds check dtype categories and tensor devices, and validate output dtypes and devices.
/// During execution, output enum variants are checked in all builds before any output handles are published.
/// Output shapes and ordinary field values are never compared with backend results, even in debug builds.
///
/// Custom kernels are opaque unless a custom optimizer recognizes their IR.
///
/// For other signatures, use an existing default body or omit `Fusion` from `#[backend_extension]`
/// and implement the trait for `Fusion<B>` manually.
///
/// # Remote
///
/// Listing `Remote` beside `Fusion` runs the extension on Burn's remote backend with no
/// handwritten code. The server only receives the operation's `CustomOpIr`, so it carries the
/// whole call:
///
/// - **inputs**: every tensor, in declaration order, an extension value's in field order;
/// - **outputs**: one per output tensor, described by the method's Fusion metadata;
/// - **scalars**: every ordinary argument in declaration order, then each extension argument's
///   variant index (enums) and ordinary fields, in declaration order, nested values inline.
///
/// Fusion builds the same IR, so an operation fused into a cached graph for the remote backend
/// reaches the server as complete as one sent alone. Two items are generated, behind both the
/// `Remote` and the `Fusion` conditions:
///
/// - the client, `impl<R: RouterChannel> Trait for BackendRouter<R>`, which the generated
///   `Fusion<B>` implementation wraps into the remote backend. The remote backend is
///   `Fusion<BackendRouter<_>>` whenever Burn's `fusion` feature is on, so `Fusion`'s condition
///   must hold there too; where it does not, the `Remote` dispatch arm is left out.
/// - `TraitHandlers`, with the trait's visibility, whose `register` adds one handler per method to
///   the server's registry for any backend implementing the trait:
///
/// ```rust,ignore
/// #[backend_extension(Cube, Fusion, Remote)]
/// pub trait AttentionBackend: Backend {
///     #[fusion(dtype = query, shape = query)]
///     fn attend(
///         query: FloatTensor<Self>,
///         #[fusion(scalar = strategy.to_code(), decode = |code| Strategy::from_code(code).expect("a strategy"))]
///         strategy: Strategy,
///         #[extension_type] kv: Kv<Self>,
///     ) -> FloatTensor<Self>;
/// }
///
/// // On the server, once per trait:
/// let mut registry = CustomOpRegistry::<Cube>::new();
/// AttentionBackendHandlers::register(&mut registry);
/// RemoteServer::new(devices).with_custom_ops(registry).serve(transport)?;
/// ```
///
/// Every ordinary argument must travel as a scalar: primitives do on their own, and anything else
/// needs `#[fusion(scalar = expression, decode = callable)]`. The callable takes the primitive the
/// expression returned, inferred from its signature, and returns the argument; a code that
/// can fail to decode is unwrapped there. A bare `#[fusion(scalar)]` decodes when the type is an
/// alias of a primitive. Anything else is a compile error under `Remote`. Extension arguments
/// derive [`ExtensionType`] with `#[extension_type(fusion, remote)]`.
///
/// The handler checks that the IR holds exactly the inputs and scalars the signature reads and
/// that the backend returns one tensor per output id; debug builds also check the client's dtypes
/// and that inputs share a device. Output shapes and ordinary output fields come from the
/// metadata, as with Fusion. Without `Fusion`, `Remote` only routes, to a handwritten
/// implementation.
///
/// `register` panics if another handler already claims one of the operation ids: two traits
/// with the same name, or two methods given the same explicit `id`, must be told apart with
/// `#[fusion(id = ..)]`. Registering the same trait twice is harmless.
///
/// A handler trusts the IR it receives. One that does not match the method's signature (fewer
/// or more inputs or scalars than it reads, an unknown enum variant index, a scalar that does not
/// decode, or an operation id with no registered handler) panics on the thread serving that
/// client's session: the session stops answering, rather than one operation returning an error.
/// Client and server must therefore be built from the same trait definition.
#[proc_macro_attribute]
pub fn backend_extension(attr: TokenStream, item: TokenStream) -> TokenStream {
    extension::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Maps structs and enums of backend tensor primitives across the `Dispatch` boundary.
///
/// Opt into `#[extension_type(fusion)]` (optionally `fusion: cfg(...)`) to support Fusion inputs
/// and outputs and generate a backend-independent `NameMetadata` type. It mirrors the struct's
/// fields or enum's variants, replacing tensors with `TensorSpec` and nested extension values with
/// their metadata. Ordinary fields are cloned and must implement `Clone + Debug`;
/// ordinary input fields must also be `Send + Sync + 'static`.
///
/// The metadata callback chooses the output variant and supplies the actual return values for
/// non-tensor fields. Fusion returns those values without waiting for execution and ignores the backend's
/// later values for those fields. The callback must calculate what a direct backend call would return;
/// otherwise enabling Fusion changes the result. Generated code does not check these field values.
/// Empty variants are supported, but each operation still needs an input tensor for its device.
///
/// Add `remote` beside `fusion`, as in `#[extension_type(fusion, remote)]`, to send the value to
/// the remote backend as part of a custom operation's IR (see [`backend_extension`]'s Remote
/// section). An enum travels as its variant index; ordinary fields travel as scalars, which
/// primitives do on their own and other types through
/// `#[extension_type(scalar = expression, decode = callable)]` on the field. The expression sees a
/// named field by its name, as a reference, and a positional one as `value`.
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
