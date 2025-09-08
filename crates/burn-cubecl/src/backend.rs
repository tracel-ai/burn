use crate::{CubeRuntime, FloatElement, IntElement, element::BoolElement, tensor::CubeTensor};
use burn_tensor::backend::{Backend, DeviceOps};
use cubecl::server::ComputeServer;
use rand::{SeedableRng, rngs::StdRng};
use std::{marker::PhantomData, sync::Mutex};

#[cfg(not(feature = "fusion"))]
use burn_ir::{BackendIr, TensorHandle};
#[cfg(not(feature = "fusion"))]
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct CubeBackend<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, F, I, BT> Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
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

    type FloatTensorPrimitive = CubeTensor<R>;
    type IntTensorPrimitive = CubeTensor<R>;
    type BoolTensorPrimitive = CubeTensor<R>;
    type QuantizedTensorPrimitive = CubeTensor<R>;

    fn name(device: &Self::Device) -> String {
        let client = R::client(device);
        format!("cubecl<{}>", R::name(&client))
    }

    fn seed(_device: &Self::Device, seed: u64) {
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

    fn memory_static_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        let client = R::client(device);
        let output = client.memory_static_allocation(input, func);
        let memory = client.memory_usage();
        println!("{memory}");
        output
    }

    fn memory_cleanup(device: &Self::Device) {
        let client = R::client(device);
        client.memory_cleanup();
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> core::fmt::Debug
    for CubeBackend<R, F, I, BT>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CubeCLBackend")
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Clone
    for CubeBackend<R, F, I, BT>
{
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Default
    for CubeBackend<R, F, I, BT>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R: cubecl::Runtime> CubeRuntime for R
where
    R::Device: DeviceOps,
{
    type CubeDevice = R::Device;
    type CubeServer = R::Server;
}

#[cfg(not(feature = "fusion"))]
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendIr
    for CubeBackend<R, F, I, BT>
{
    type Handle = CubeTensor<R>;

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
