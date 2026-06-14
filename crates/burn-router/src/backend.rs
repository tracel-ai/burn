use super::{RouterChannel, RouterClient, RouterTensor, get_client};
use alloc::{format, string::String};
use burn_backend::{Backend, BackendTypes, DType, ExecutionError, UnimplementedTensorPrimitive};
use core::marker::PhantomData;

/// A backend that forwards the tensor operations to the appropriate backend (given multiple backends).
pub struct BackendRouter<R: RouterChannel> {
    r: PhantomData<R>,
}

impl<R: RouterChannel> core::fmt::Debug for BackendRouter<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("router"))
    }
}

impl<R: RouterChannel> Clone for BackendRouter<R> {
    fn clone(&self) -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RouterChannel> Default for BackendRouter<R> {
    fn default() -> Self {
        Self { r: PhantomData }
    }
}

impl<R: RouterChannel> BackendTypes for BackendRouter<R> {
    type Device = R::Device;

    type FloatTensorPrimitive = RouterTensor<R::Client>;
    type IntTensorPrimitive = RouterTensor<R::Client>;
    type BoolTensorPrimitive = RouterTensor<R::Client>;
    type QuantizedTensorPrimitive = RouterTensor<R::Client>;
    type ComplexTensorPrimitive = UnimplementedTensorPrimitive<R::Client>;
}

impl<R: RouterChannel> Backend for BackendRouter<R> {
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

    fn dtype_usage(device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        let client = get_client::<R>(device);
        client.dtype_usage(dtype)
    }

    fn device_count(_: u16) -> usize {
        // This is what was there before, not sure if it's actually correct
        1
    }

    fn flush(device: &Self::Device) {
        let client = get_client::<R>(device);
        client.flush();
    }
}
