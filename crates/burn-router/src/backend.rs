use super::{RouterTensor, RunnerChannel, RunnerClient, get_client};
use alloc::{format, string::String};
use burn_tensor::{
    DType,
    backend::{Backend, ExecutionError},
    quantization::{QTensorPrimitive, QuantScheme},
};
use core::marker::PhantomData;

/// A backend that forwards the tensor operations to the appropriate backend (given multiple backends).
pub struct BackendRouter<R: RunnerChannel> {
    r: PhantomData<R>,
}

impl<R: RunnerChannel> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: RunnerChannel> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerChannel> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RunnerClient> QTensorPrimitive for RouterTensor<R> {
    fn scheme(&self) -> &QuantScheme {
        if let DType::QFloat(scheme) = &self.dtype {
            scheme
        } else {
            // TODO: maybe `tensor.scheme()` should return an option
            panic!("Expected quantized float dtype, got {:?}", self.dtype)
        }
    }
}

impl<R: RunnerChannel> Backend for BackendRouter<R> {
    type Device = R::Device;

    type FloatTensorPrimitive = RouterTensor<R::Client>;

    type FloatElem = R::FloatElem;

    type IntTensorPrimitive = RouterTensor<R::Client>;

    type IntElem = R::IntElem;

    type BoolTensorPrimitive = RouterTensor<R::Client>;

    type BoolElem = R::BoolElem;

    type QuantizedTensorPrimitive = RouterTensor<R::Client>;

    fn name(device: &Self::Device) -> String {
        format!("router<{}>", R::name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        let client = get_client::<R>(device);
        client.seed(seed);
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = get_client::<R>(device);
        client.sync()
    }

    fn supports_dtype(device: &Self::Device, dtype: DType) -> bool {
        let client = get_client::<R>(device);
        client.supports_dtype(dtype)
    }
}
