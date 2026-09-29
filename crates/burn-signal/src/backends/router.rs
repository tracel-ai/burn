use burn_core::backend::{
    ir::{BackendIr, OperationIr, OperationOutput},
    tensor::FloatTensor,
};
use burn_router::{BackendRouter, CustomOpRegistry, RouterChannel, RouterClient};

use crate::{SignalOps, custom};

/// Register signal FFT handlers for a remote server or captured-graph interpreter.
///
/// Enable the `router` feature on the server and call this before passing the registry
/// to `TensorInterpreter::with_custom_ops` or `RemoteServerBuilder::custom_ops`.
/// The server backend must also have its corresponding `burn-signal` feature enabled.
pub fn register_fft_ops<B: BackendIr + SignalOps>(registry: &mut CustomOpRegistry<B>) {
    registry.register(custom::RFFT, |handles, desc, _device| {
        custom::execute::<B>(handles, desc)
    });
    registry.register(custom::IRFFT, |handles, desc, _device| {
        custom::execute::<B>(handles, desc)
    });
}

impl<C: RouterChannel> SignalOps for BackendRouter<C> {
    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let client = signal.client.clone();
        let desc = custom::rfft(signal.into_ir(), dim, n, || client.create_empty_handle());
        let [real, imag] = client.register(OperationIr::Custom(desc)).outputs();
        (real, imag)
    }

    fn irfft(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        let client = real.client.clone();
        let desc = custom::irfft(real.into_ir(), imag.into_ir(), dim, n, || {
            client.create_empty_handle()
        });
        client.register(OperationIr::Custom(desc)).output()
    }
}
