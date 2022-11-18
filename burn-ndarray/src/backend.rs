use super::element::NdArrayElement;
use super::NdArrayTensor;
use burn_tensor::backend::Backend;
use burn_tensor::Data;
use burn_tensor::{Distribution, Shape};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Mutex;

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

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
    _e: E,
}

impl<E: NdArrayElement> Backend for NdArrayBackend<E> {
    type Device = NdArrayDevice;
    type Elem = E;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = NdArrayBackend<f32>;
    type IntegerBackend = NdArrayBackend<i64>;
    type TensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        _device: Self::Device,
    ) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        _device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D> {
        NdArrayTensor::from_data(data)
    }

    fn ad_enabled() -> bool {
        false
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D> {
        let mut seed = SEED.lock().unwrap();
        let mut rng: StdRng = match seed.as_ref() {
            Some(rng) => rng.clone(),
            None => StdRng::from_entropy(),
        };
        let tensor = Self::from_data(Data::random(shape, distribution, &mut rng), device);
        *seed = Some(rng);
        tensor
    }

    fn name() -> String {
        "ndarray".to_string()
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }
}
