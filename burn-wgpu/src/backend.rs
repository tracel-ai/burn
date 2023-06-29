use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    element::{FloatElement, IntElement},
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Wgpu backend.
#[derive(Debug, Default, Clone)]
pub struct WgpuBackend<G: GraphicsApi, F: FloatElement, I: IntElement> {
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<G: GraphicsApi + 'static, F: FloatElement, I: IntElement> Backend for WgpuBackend<G, F, I> {
    type Device = WgpuDevice;
    type FullPrecisionBackend = WgpuBackend<G, f32, i32>;

    type FullPrecisionElem = f32;
    type FloatElem = F;
    type IntElem = I;

    type TensorPrimitive<const D: usize> = WgpuTensor<F, D>;
    type IntTensorPrimitive<const D: usize> = WgpuTensor<I, D>;
    type BoolTensorPrimitive<const D: usize> = WgpuTensor<u32, D>;

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
