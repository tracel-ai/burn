use crate::{BackendTypes, ops::ComplexTensorOps, tensor::FloatTensor};

/// Complex tensor primitive type used by the backend.
pub type ComplexTensor<B> = <B as BackendTypes>::ComplexTensorPrimitive;

/// Backend for complex tensor operations. This trait can act either as an extension or a wrapper
/// around a standard `Backend` implementation. For backends that don't yet natively support complex
/// operations, a default implementation exists for any backend that sets the complex primitive to
/// UnimplementedTensorPrimitive.
pub trait ComplexTensorBackend:
    ComplexTensorOps<Self> + Sized + BackendTypes + Send + Sync + 'static
{
}
