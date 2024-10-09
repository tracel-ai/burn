use crate::{
    backend::Backend,
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
    quantization::QuantizationScheme,
    Shape,
};
use alloc::vec::Vec;

/// A tensor representation containing a reference to a tensor resource with a given shape.
pub struct TensorHandle<H> {
    /// The type that can be used to point to a tensor of any kind.
    pub handle: H,
    /// The shape associated to the tensor.
    pub shape: Shape,
}

/// Backend extension trait that allows an existing [backend](Backend) to use the Burn tensor representation
/// for compilation purpose or other...
pub trait ReprBackend: Backend {
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;

    /// Convert a [handle](ReprBackend::Handle) to a [float tensor](Backend::FloatTensorPrimitive).
    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self>;
    /// Convert a [handle](ReprBackend::Handle) to a [quantized tensor](Backend::QuantizedTensorPrimitive).
    fn quantized_tensor(
        handles: Vec<TensorHandle<Self::Handle>>,
        scheme: QuantizationScheme,
    ) -> QuantizedTensor<Self>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](ReprBackend::Handle).
    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle;
    /// Convert a [quantized tensor](Backend::QuantizedTensorPrimitive) to a [handle](ReprBackend::Handle).
    /// A quantized tensor has multiple handles for the tensor itself and the quantization parameters.
    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Vec<Self::Handle>;
}
