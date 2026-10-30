use super::{RouterChannel, RouterClient, RouterGraph, RouterTensor, get_client};
use alloc::{format, string::String};
use burn_backend::{
    Backend, BackendGraph, BackendTypes, DType, ExecutionError, ExecutionMode, ExecutionModeToken,
    ExecutionStatistics, ProfileDuration, ProfileOptions, ProfileToken, profile_with_tokens,
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

    type GraphPrimitive = RouterGraph<R::Client>;
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
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        // The interpreter is where the window opens; the flush travels to it
        // with the close, for the queue of the backend behind it to drain.
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

    fn profile_abandon(device: &Self::Device, token: ProfileToken) {
        let client = get_client::<R>(device);
        client.profile_abandon(token);
    }

    fn execution_mode_open(
        device: &Self::Device,
        mode: ExecutionMode,
    ) -> Result<Option<ExecutionModeToken>, ExecutionError> {
        let client = get_client::<R>(device);
        client.execution_mode_open(mode)
    }

    fn execution_mode_statistics(
        device: &Self::Device,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let client = get_client::<R>(device);
        client.execution_mode_statistics(token)
    }

    fn execution_mode_close(
        device: &Self::Device,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let client = get_client::<R>(device);
        client.execution_mode_close(token)
    }

    fn dtype_usage(device: &Self::Device, dtype: DType) -> burn_backend::DTypeUsageSet {
        let client = get_client::<R>(device);
        client.dtype_usage(dtype)
    }

    fn device_count(_: u16) -> usize {
        // This is what was there before, not sure if it's actually correct
        1
    }

    fn flush(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = get_client::<R>(device);
        client.flush();
        Ok(())
    }

    fn graph_prepare(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = get_client::<R>(device);
        client.graph_prepare()
    }

    // The interpreter records the graph on its own backend, so the capture opens and closes
    // there, in order with the operations around it.
    fn graph_start_capture(device: &Self::Device) -> Result<(), ExecutionError> {
        let client = get_client::<R>(device);
        client.graph_start_capture()
    }

    fn graph_stop_capture(device: &Self::Device) -> Result<BackendGraph<Self>, ExecutionError> {
        let client = get_client::<R>(device);
        let id = client.graph_stop_capture()?;
        Ok(RouterGraph::new(id, client))
    }

    unsafe fn graph_replay(
        _device: &Self::Device,
        graph: &BackendGraph<Self>,
    ) -> Result<(), ExecutionError> {
        // Safety: forwarded verbatim from this method's own contract.
        unsafe { graph.replay() }
    }
}
