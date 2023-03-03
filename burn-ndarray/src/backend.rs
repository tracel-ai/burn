use alloc::string::String;
use core::marker::PhantomData;

use crate::element::NdArrayElement;
use crate::NdArrayTensor;

use burn_tensor::backend::Backend;

use rand::{rngs::StdRng, SeedableRng};

#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(not(feature = "std"))]
use burn_common::stub::Mutex;

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

#[derive(Clone, Copy, Debug)]
pub enum NdArrayDevice {
    Cpu,
}

impl Default for NdArrayDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct NdArrayBackend<E> {
    phantom: PhantomData<E>,
}

impl<E: NdArrayElement> Backend for NdArrayBackend<E> {
    type Device = NdArrayDevice;
    type FloatElem = E;
    type IntElem = i64;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = NdArrayBackend<f32>;
    type IntegerBackend = NdArrayBackend<i64>;
    type TensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
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
