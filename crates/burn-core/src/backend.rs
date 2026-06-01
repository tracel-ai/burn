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
    fn map_type<F>(self, map_kind: F, checkpointing: Option<CheckpointingStrategy>) -> Self::Target
    where
        F: Fn(BackendTensor<B>) -> DispatchTensorKind;
}
