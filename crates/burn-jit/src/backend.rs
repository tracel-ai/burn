use crate::{codegen::Compiler, tensor::JitTensor, PrecisionBridge, Runtime};
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct JitBackend<R: Runtime> {
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Backend for JitBackend<R> {
    type Device = R::Device;

    type FullPrecisionBridge = PrecisionBridge<R::FullPrecisionRuntime>;
    type FloatElem = <R::Compiler as Compiler>::Float;
    type IntElem = <R::Compiler as Compiler>::Int;

    type FloatTensorPrimitive<const D: usize> = JitTensor<R, Self::FloatElem, D>;
    type IntTensorPrimitive<const D: usize> = JitTensor<R, Self::IntElem, D>;
    type BoolTensorPrimitive<const D: usize> = JitTensor<R, u32, D>;

    fn name() -> String {
        format!("jit<{}>", R::name())
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

impl<R: Runtime> core::fmt::Debug for JitBackend<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("JitBackend {{ runtime: {}}}", R::name()))
    }
}

impl<R: Runtime> Clone for JitBackend<R> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<R: Runtime> Default for JitBackend<R> {
    fn default() -> Self {
        Self::new()
    }
}
