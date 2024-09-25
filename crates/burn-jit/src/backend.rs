use crate::{
    tensor::{JitTensor, QJitTensor},
    FloatElement, IntElement, JitRuntime, PrecisionBridge,
};
use burn_tensor::backend::{Backend, DeviceOps, SyncType};
use cubecl::server::ComputeServer;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct JitBackend<R: JitRuntime, F: FloatElement, I: IntElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
}

impl<R, F, I> Backend for JitBackend<R, F, I>
where
    R: JitRuntime,
    R::Server: ComputeServer,
    R::Device: burn_tensor::backend::DeviceOps,
    F: FloatElement,
    I: IntElement,
{
    type Device = R::Device;

    type FullPrecisionBridge = PrecisionBridge<R, f32, i32>;
    type FloatElem = F;
    type IntElem = I;

    type FloatTensorPrimitive = JitTensor<R, Self::FloatElem>;
    type IntTensorPrimitive = JitTensor<R, Self::IntElem>;
    type BoolTensorPrimitive = JitTensor<R, u32>;
    type QuantizedTensorPrimitive = QJitTensor<R, Self::FloatElem, Self::IntElem>;
    type QuantizedEncoding = u32;

    fn name() -> String {
        format!("jit<{}>", R::name())
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }

    fn ad_enabled() -> bool {
        false
    }

    fn sync(device: &Self::Device, sync_type: SyncType) {
        let sync = match sync_type {
            SyncType::Flush => cubecl::client::SyncType::Flush,
            SyncType::Wait => cubecl::client::SyncType::Wait,
        };
        let client = R::client(device);
        client.sync(sync);
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> core::fmt::Debug for JitBackend<R, F, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("JitBackend {{ runtime: {}}}", R::name()))
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> Clone for JitBackend<R, F, I> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> Default for JitBackend<R, F, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: cubecl::Runtime> JitRuntime for R
where
    R::Device: DeviceOps,
{
    type JitDevice = R::Device;
    type JitServer = R::Server;
}
