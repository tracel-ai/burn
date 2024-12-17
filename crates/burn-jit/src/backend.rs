use crate::{element::BoolElement, tensor::JitTensor, FloatElement, IntElement, JitRuntime};
use burn_tensor::backend::{Backend, DeviceOps};
use cubecl::server::ComputeServer;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

#[cfg(not(feature = "fusion"))]
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
    repr::{ReprBackend, TensorHandle},
};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct JitBackend<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, F, I, BT> Backend for JitBackend<R, F, I, BT>
where
    R: JitRuntime,
    R::Server: ComputeServer,
    R::Device: burn_tensor::backend::DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Device = R::Device;

    type FloatElem = F;
    type IntElem = I;
    type BoolElem = BT;

    type FloatTensorPrimitive = JitTensor<R>;
    type IntTensorPrimitive = JitTensor<R>;
    type BoolTensorPrimitive = JitTensor<R>;
    type QuantizedTensorPrimitive = JitTensor<R>;
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

    fn sync(device: &Self::Device) {
        let client = R::client(device);
        futures_lite::future::block_on(client.sync());
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> core::fmt::Debug
    for JitBackend<R, F, I, BT>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("JitBackend {{ runtime: {}}}", R::name()))
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Clone
    for JitBackend<R, F, I, BT>
{
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Default
    for JitBackend<R, F, I, BT>
{
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

#[cfg(not(feature = "fusion"))]
impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> ReprBackend
    for JitBackend<R, F, I, BT>
{
    type Handle = JitTensor<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }
}
