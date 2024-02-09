use crate::{codegen::Compiler, tensor::JitTensor, Runtime};
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
pub struct JitBackend<R: Runtime> {
    _b: PhantomData<R>,
}

impl<R: Runtime> core::fmt::Debug for JitBackend<R> {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<R: Runtime> Clone for JitBackend<R> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<R: Runtime> Default for JitBackend<R> {
    fn default() -> Self {
        todo!()
    }
}

impl<R: Runtime> Backend for JitBackend<R> {
    type Device = R::Device;
    type FullPrecisionBackend = JitBackend<R::FullPrecisionRuntime>;

    type FullPrecisionElem = f32;
    type FloatElem = <R::Compiler as Compiler>::Float;
    type IntElem = <R::Compiler as Compiler>::Int;

    type FloatTensorPrimitive<const D: usize> = JitTensor<R, Self::FloatElem, D>;
    type IntTensorPrimitive<const D: usize> = JitTensor<R, Self::IntElem, D>;
    type BoolTensorPrimitive<const D: usize> = JitTensor<R, u32, D>;

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

    fn sync(device: &Self::Device) {
        let client = R::client(device);
        client.sync();
    }
}
