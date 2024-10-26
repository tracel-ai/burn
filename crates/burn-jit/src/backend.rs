use crate::{
    tensor::{JitTensor, QJitTensor},
    FloatElement, IntElement, JitRuntime, PrecisionBridge,
};
use burn_tensor::backend::{Backend, DeviceOps};
use cubecl::server::ComputeServer;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

#[cfg(not(feature = "fusion"))]
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
    quantization::QuantizationScheme,
    repr::{HandleKind, QuantizedKind, ReprBackend, TensorHandle},
};

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

    fn sync(device: &Self::Device) {
        let client = R::client(device);
        futures_lite::future::block_on(client.sync());
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

#[cfg(not(feature = "fusion"))]
impl<R: JitRuntime, F: FloatElement, I: IntElement> ReprBackend for JitBackend<R, F, I> {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        match handle.handle {
            HandleKind::Float(handle) => handle,
            _ => panic!("Expected float handle, got {}", handle.handle.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        match handle.handle {
            HandleKind::Int(handle) => handle,
            _ => panic!("Expected int handle, got {}", handle.handle.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        match handle.handle {
            HandleKind::Bool(handle) => handle,
            _ => panic!("Expected bool handle, got {}", handle.handle.name()),
        }
    }

    fn quantized_tensor(
        handles: QuantizedKind<TensorHandle<Self::Handle>>,
        _scheme: QuantizationScheme,
    ) -> QuantizedTensor<Self> {
        let handle = handles.tensor.handle;
        match handle {
            HandleKind::Quantized(handle) => handle,
            _ => panic!("Expected quantized handle, got {}", handle.name()),
        }
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        HandleKind::Float(tensor)
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        HandleKind::Int(tensor)
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        HandleKind::Bool(tensor)
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> QuantizedKind<Self::Handle> {
        QuantizedKind {
            tensor: HandleKind::Quantized(tensor),
            // The quantized tensor primitive already encapsulates the required quantization
            // parameters so we set the scale as an empty handle (unused).
            scale: HandleKind::Empty,
            offset: None,
        }
    }
}
