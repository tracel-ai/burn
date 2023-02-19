use super::element::NdArrayElement;
use super::NdArrayTensor;
use burn_tensor::backend::Backend;
use core::marker::PhantomData;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use spin::Mutex;

use const_random::const_random;

extern crate alloc;
use alloc::string::String;

pub(crate) static SEED: Mutex<Option<SmallRng>> = Mutex::new(None);

pub const GENERATED_SEED: u64 = const_random!(u64);

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
    type Elem = E;
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
        let rng = SmallRng::seed_from_u64(seed);
        let mut seed = SEED.lock();
        *seed = Some(rng);
    }
}
