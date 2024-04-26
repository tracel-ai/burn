use crate::{
    backend::Backend,
    ops::{BoolTensor, FloatTensor, IntTensor},
    Shape,
};

/// Backend extension trait that allows an existing [backend](Backend) to use the Burn tensor representation
/// for compilation purpose or other...
pub trait ReprBackend: Backend {
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;

    /// Convert a [handle](ReprBackend::Handle) to a [float tensor](Backend::FloatTensorPrimitive).
    fn float_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> FloatTensor<Self, D>;
    /// Convert a [handle](ReprBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> IntTensor<Self, D>;
    /// Convert a [handle](ReprBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> BoolTensor<Self, D>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn float_tensor_handle<const D: usize>(tensor: FloatTensor<Self, D>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn int_tensor_handle<const D: usize>(tensor: IntTensor<Self, D>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn bool_tensor_handle<const D: usize>(tensor: BoolTensor<Self, D>) -> Self::Handle;
}
