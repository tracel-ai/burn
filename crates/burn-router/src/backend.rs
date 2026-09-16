use super::{RouterChannel, RouterClient, RouterTensor, get_client};
use alloc::{format, string::String};
use burn_backend::{
    Backend, BackendTypes, DType, ExecutionError, ProfileDuration, ProfileOptions, ProfileToken,
    profile_with_tokens,
};
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

    type GraphPrimitive = burn_backend::GraphUnsupported;
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

    fn profile<O: Send + 'static>(
        device: &Self::Device,
        name: &str,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        // The interpreter is where the window opens; the flush travels to it
        // with the close, for the queue of the backend behind it to drain.
        // The name stops here: the split window carries none.
        let _ = name;
        profile_with_tokens::<Self, O>(device, options, func)
    }

    fn profile_start(device: &Self::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        let client = get_client::<R>(device);
        client.profile_start()
    }

    fn profile_end(
        device: &Self::Device,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        let client = get_client::<R>(device);
        client.profile_end(token, options)
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
