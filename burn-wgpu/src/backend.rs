use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    element::{FloatElement, IntElement},
    tensor::WGPUTensor,
    GraphicsAPI, WGPUDevice,
};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

#[derive(Debug, Default, Clone)]
pub struct WGPUBackend<G: GraphicsAPI, F: FloatElement, I: IntElement> {
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<G: GraphicsAPI + 'static, F: FloatElement, I: IntElement> Backend for WGPUBackend<G, F, I> {
    type Device = WGPUDevice;
    type FullPrecisionBackend = WGPUBackend<G, f32, i32>;

    type FullPrecisionElem = f32;

    type TensorPrimitive<const D: usize> = WGPUTensor<D>;

    type FloatElem = F;

    type IntTensorPrimitive<const D: usize> = WGPUTensor<D>;

    type IntElem = I;

    type BoolTensorPrimitive<const D: usize> = WGPUTensor<D>;

    fn name() -> String {
        String::from("wgpu")
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }

    fn ad_enabled() -> bool {
        false
    }
}
