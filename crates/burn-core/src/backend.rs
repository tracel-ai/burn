#[cfg(feature = "ir")]
pub use burn_ir as ir;

pub use burn_backend::*;
pub use burn_backend_extension::{ExtensionType, backend_extension};

// Dispatch backend extension types
pub use burn_dispatch::{backend::*, device::*, tensor::*};
// Re-export backends (e.g., Cuda)
pub use burn_dispatch::backends::*;

/// A trait to map custom structs and enums of tensor primitives across the [`Dispatch`] boundary, in
/// both directions.
///
/// This trait cooperates with the [`#[backend_extension]`](backend_extension) macro. When an extension
/// operation returns such a type, [`map_to_dispatch`](Self::map_to_dispatch) wraps each internal tensor
/// into a [`DispatchTensor`]; when it takes one as an input, [`map_from_dispatch`](Self::map_from_dispatch)
/// reconstructs the concrete value and [`routing_tensor`](Self::routing_tensor) /
/// [`routing_float_tensor`](Self::routing_float_tensor) locate a tensor for dispatch routing and
/// backend and autodiff-context selection. All tensor fields participating in one operation must
/// share the routing tensor's context. Nested `#[extension_type]` fields are traversed recursively.
///
/// Implementations are generated automatically using `#[derive(ExtensionType)]`.
pub trait ExtensionType<B: Backend> {
    /// The target struct layout where all internal concrete backend tensors are transformed
    /// into [`DispatchTensor`]s.
    type Target;

    /// Transforms the internal fields of the struct by applying a backend-specific wrapping closure.
    ///
    /// # Arguments
    ///
    /// * `map_kind` - A closure provided by the dispatch macro that knows how to map a backend-agnostic
    ///   [`BackendTensor`] variant into the correct [`DispatchTensorKind`] variant (e.g., `Wgpu`, `Cuda`, `Cpu`).
    /// * `autodiff` - The semantic autodiff backend context to attach to each [`DispatchTensor`].
    ///
    /// # Returns
    ///
    /// A new instance of the struct mapped to the [`Dispatch`] backend.
    fn map_to_dispatch<F>(self, map_kind: F, autodiff: DispatchAutodiffContext) -> Self::Target
    where
        F: Fn(BackendTensor<B>) -> DispatchTensorKind;

    /// Reconstruct the concrete `Struct<B>` from its dispatch form `Struct<Dispatch>`.
    ///
    /// This is the inverse of [`map_to_dispatch`](Self::map_to_dispatch), used when a custom struct is passed as
    /// an **input** to a backend extension operation. The dispatch glue has already selected the
    /// target backend `B`; `unwrap_kind` pulls the matching [`BackendTensor`] out of each field's
    /// [`DispatchTensorKind`], and the derived impl calls the right accessor (`.float()`, `.int()`,
    /// ...) per field to recover the concrete primitive.
    ///
    /// # Arguments
    ///
    /// * `unwrap_kind` - A closure provided by the dispatch macro that validates the tensor's
    ///   autodiff context and unwraps its [`DispatchTensorKind`] into the [`BackendTensor`] for the
    ///   selected backend `B`, panicking on a backend mismatch.
    fn map_from_dispatch<F>(target: Self::Target, unwrap_kind: F) -> Self
    where
        F: Fn(DispatchTensor) -> BackendTensor<B>;

    /// Return a tensor of the dispatch form to use for routing, or `None` if this value
    /// currently holds no tensor (e.g. an enum on a tensor-less variant).
    ///
    /// A struct/enum input carries no top-level [`DispatchTensor`] of its own, so the dispatch glue
    /// uses this to read the runtime backend tag (`.kind`) and autodiff context. All other tensor
    /// fields must carry the same context; dispatch validates them while mapping the value to its
    /// concrete backend. This lookup recurses into nested `#[extension_type]` fields.
    fn routing_tensor(target: &Self::Target) -> Option<&DispatchTensor>;

    /// Like [`routing_tensor`](Self::routing_tensor) but returns only a *float* tensor, or `None` if
    /// there is none.
    ///
    /// The dispatch glue prefers a float routing tensor because active float presence decides
    /// whether the operation needs an autodiff backend. The glue falls back to
    /// [`routing_tensor`](Self::routing_tensor) only when no float tensor exists anywhere in the
    /// inputs.
    fn routing_float_tensor(target: &Self::Target) -> Option<&DispatchTensor>;
}
