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
    fn float_tensor(handle: Self::Handle, shape: Shape) -> FloatTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor(handle: Self::Handle, shape: Shape) -> IntTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor(handle: Self::Handle, shape: Shape) -> BoolTensor<Self>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle;
}
