use crate::element::{FloatNdArrayElement, QuantElement};
use crate::PrecisionBridge;
use crate::{NdArrayQTensor, NdArrayTensor};
use alloc::string::String;
use burn_common::stub::Mutex;
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
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
