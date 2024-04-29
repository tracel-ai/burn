use crate::NdArrayTensor;
use crate::{element::FloatNdArrayElement, PrecisionBridge};
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
pub struct NdArray<E = f32> {
    phantom: PhantomData<E>,
}

impl<E: FloatNdArrayElement> Backend for NdArray<E> {
    type Device = NdArrayDevice;
    type FullPrecisionBridge = PrecisionBridge<f32>;

    type FloatTensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type FloatElem = E;

    type IntTensorPrimitive<const D: usize> = NdArrayTensor<i64, D>;
    type IntElem = i64;

    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;

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
