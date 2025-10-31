use crate::rand::NdArrayRng;
use crate::{NdArrayQTensor, NdArrayTensor};
use crate::{
    SharedArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
};
use alloc::string::String;
use burn_common::stub::Mutex;
use burn_ir::{BackendIr, HandleKind, TensorHandle};
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use core::marker::PhantomData;
use rand::SeedableRng;

pub(crate) static SEED: Mutex<Option<NdArrayRng>> = Mutex::new(None);

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NdArrayDevice {
    /// The CPU device.
    #[default]
    Cpu,
}

impl DeviceOps for NdArrayDevice {}

impl burn_common::device::Device for NdArrayDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Cpu
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

/// Tensor backend that uses the [ndarray](ndarray) crate for executing tensor operations.
///
/// This backend is compatible with CPUs and can be compiled for almost any platform, including
/// `wasm`, `arm`, and `x86`.
#[derive(Clone, Copy, Default, Debug)]
pub struct NdArray<E = f32, I = i64, Q = i8>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    _e: PhantomData<E>,
    _i: PhantomData<I>,
    _q: PhantomData<Q>,
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> Backend for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Device = NdArrayDevice;

    type FloatTensorPrimitive = NdArrayTensor;
    type FloatElem = E;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = I;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("ndarray")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        let rng = NdArrayRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> BackendIr for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
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

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        match handle.handle {
            HandleKind::Quantized(handle) => handle,
            _ => panic!("Expected quantized handle, got {}", handle.handle.name()),
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

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        HandleKind::Quantized(tensor)
    }
}
