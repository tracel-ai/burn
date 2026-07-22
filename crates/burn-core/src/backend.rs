#[cfg(feature = "ir")]
pub use burn_ir as ir;

pub use burn_backend::*;
pub use burn_backend_extension::*;

// Dispatch backend extension types
pub use burn_dispatch::{backend::*, device::*, tensor::*};
// Re-export backends (e.g., Cuda)
pub use burn_dispatch::backends::*;

/// A trait to allow mapping custom backend-specific structures into their generic dispatch equivalents.
///
/// This trait is designed to cooperate with the [`#[backend_extension]`](backend_extension) macro. When an
/// extension operation returns a custom struct containing multiple tensor primitives rather than a single
/// output tensor primitive, this trait provides the mechanism to traverse that struct and recursively wrap
/// each internal field into a [`DispatchTensor`].
///
/// Implementations of this trait are generated automatically using `#[derive(ExtensionType)]`.
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
    /// * `checkpointing` - The active tensor recording/checkpointing strategy to attach to the [`DispatchTensor`].
    ///
    /// # Returns
    ///
    /// A new instance of the struct mapped to the [`Dispatch`] backend.
    fn map_to_dispatch<F>(
        self,
        map_kind: F,
        checkpointing: Option<CheckpointingStrategy>,
    ) -> Self::Target
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
    /// * `unwrap_kind` - A closure provided by the dispatch macro that unwraps a
    ///   [`DispatchTensorKind`] into the [`BackendTensor`] for the selected backend `B`, panicking
    ///   on a backend mismatch (which the dispatch layer guarantees never happens).
    fn map_from_dispatch<F>(target: Self::Target, unwrap_kind: F) -> Self
    where
        F: Fn(DispatchTensorKind) -> BackendTensor<B>;

    /// Return a representative tensor of the dispatch form, of any kind, or `None` if this value
    /// currently holds no tensor (e.g. an enum on a tensor-less variant).
    ///
    /// A struct/enum input carries no top-level [`DispatchTensor`] of its own, so the dispatch glue
    /// uses this to read the runtime backend tag (`.kind`) and propagate the autodiff checkpointing
    /// strategy (`.checkpointing`). Recurses into nested `#[extension_type]` fields.
    fn dispatch_repr(target: &Self::Target) -> Option<&DispatchTensor>;

    /// Like [`dispatch_repr`](Self::dispatch_repr) but returns only a *float* tensor, or `None` if
    /// there is none.
    ///
    /// The dispatch glue prefers a float representative because floats are the tensors that carry
    /// autodiff tracking (so this decides whether the op routes to the autodiff arm) and the
    /// checkpointing strategy. It falls back to [`dispatch_repr`](Self::dispatch_repr) only when no
    /// float tensor exists anywhere in the inputs.
    fn dispatch_float_repr(target: &Self::Target) -> Option<&DispatchTensor>;
}
