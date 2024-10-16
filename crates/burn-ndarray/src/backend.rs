use crate::element::{FloatNdArrayElement, QuantElement};
use crate::PrecisionBridge;
use crate::{NdArrayQTensor, NdArrayTensor};
use alloc::string::String;
use burn_common::stub::Mutex;
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_tensor::quantization::QuantizationScheme;
use burn_tensor::repr::{QuantizedKind, ReprBackend, TensorHandle};
use core::marker::PhantomData;
use rand::{rngs::StdRng, SeedableRng};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdArrayDevice {
    /// The CPU device.
    Cpu,
}

impl DeviceOps for NdArrayDevice {
    fn id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            NdArrayDevice::Cpu => DeviceId::new(0, 0),
        }
    }
}

impl Default for NdArrayDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Tensor backend that uses the [ndarray](ndarray) crate for executing tensor operations.
///
/// This backend is compatible with CPUs and can be compiled for almost any platform, including
/// `wasm`, `arm`, and `x86`.
#[derive(Clone, Copy, Default, Debug)]
pub struct NdArray<E = f32, Q = i8> {
    _e: PhantomData<E>,
    _q: PhantomData<Q>,
}

impl<E: FloatNdArrayElement, Q: QuantElement> Backend for NdArray<E, Q> {
    type Device = NdArrayDevice;
    type FullPrecisionBridge = PrecisionBridge<f32>;

    type FloatTensorPrimitive = NdArrayTensor<E>;
    type FloatElem = E;

    type IntTensorPrimitive = NdArrayTensor<i64>;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor<bool>;

    type QuantizedTensorPrimitive = NdArrayQTensor<Q>;
    type QuantizedEncoding = Q;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        String::from("ndarray")
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }
}

/// Handle which points to a backend tensor primitive kind.
// NOTE: could possibly be moved to tensor representation if used across other backends.
#[derive(Clone, Debug)]
pub enum HandleKind<B: Backend> {
    /// Float tensor handle.
    Float(B::FloatTensorPrimitive),
    /// Int tensor handle.
    Int(B::IntTensorPrimitive),
    /// Bool tensor handle.
    Bool(B::BoolTensorPrimitive),
    /// Quantized tensor handle.
    Quantized(B::QuantizedTensorPrimitive),
    /// Empty handle (used as a dummy representation).
    Empty,
}

impl<B: Backend> HandleKind<B> {
    fn dtype_str(&self) -> &str {
        match self {
            HandleKind::Float(_) => "float",
            HandleKind::Int(_) => "int",
            HandleKind::Bool(_) => "bool",
            HandleKind::Quantized(_) => "quantized",
            HandleKind::Empty => unreachable!(), // should not happen
        }
    }
}

impl<E: FloatNdArrayElement, Q: QuantElement> ReprBackend for NdArray<E, Q> {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        match handle.handle {
            HandleKind::Float(handle) => handle,
            _ => panic!("Expected float handle, got {}", handle.handle.dtype_str()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        match handle.handle {
            HandleKind::Int(handle) => handle,
            _ => panic!("Expected int handle, got {}", handle.handle.dtype_str()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        match handle.handle {
            HandleKind::Bool(handle) => handle,
            _ => panic!("Expected bool handle, got {}", handle.handle.dtype_str()),
        }
    }

    fn quantized_tensor(
        handles: QuantizedKind<TensorHandle<Self::Handle>>,
        _scheme: QuantizationScheme,
    ) -> QuantizedTensor<Self> {
        let handle = handles.tensor.handle;
        match handle {
            HandleKind::Quantized(handle) => handle,
            _ => panic!("Expected quantized handle, got {}", handle.dtype_str()),
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
