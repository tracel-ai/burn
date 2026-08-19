use burn_core::backend::tensor::FloatTensor;
use burn_fusion::{
    ExecutionError, Fusion, FusionBackend, FusionRuntime,
    custom::{CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId},
};

use crate::{SignalOps, custom};

#[derive(Debug)]
struct Fft<B> {
    desc: CustomOpIr,
    _backend: core::marker::PhantomData<B>,
}

impl<B: FusionBackend + SignalOps> Operation<B::FusionRuntime> for Fft<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        custom::execute::<B>(handles, &self.desc);
        Ok(())
    }
}

impl<B: FusionBackend + SignalOps> SignalOps for Fusion<B> {
    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let client = signal.client.clone();
        let desc = custom::rfft(signal.into_ir(), dim, n, || client.create_empty_handle());
        let [real, imag] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                Fft::<B> {
                    desc,
                    _backend: core::marker::PhantomData,
                },
            )
            .outputs();
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
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                Fft::<B> {
                    desc,
                    _backend: core::marker::PhantomData,
                },
            )
            .output()
    }
}
