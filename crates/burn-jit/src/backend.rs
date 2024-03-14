use crate::{codegen::Compiler, tensor::JitTensor, Runtime};
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};
use burn_tensor::{DynData, DynRankData};
use crate::gpu::Elem;
use crate::tensor::DynJitTensor;

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Generic tensor backend that can be compiled just-in-time to any shader runtime
#[derive(new)]
pub struct JitBackend<R: Runtime> {
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Backend for JitBackend<R>
where
    R::FullPrecisionRuntime: Runtime<Server = R::Server, Channel = R::Channel, Device = R::Device>
{
    type Device = R::Device;
    type FullPrecisionBackend = JitBackend<R::FullPrecisionRuntime>;

    type FullPrecisionElem = f32;
    type FloatTensorPrimitive<const D: usize> = JitTensor<R, Self::FloatElem, D>;
    type FloatElem = <R::Compiler as Compiler>::Float;

    type IntTensorPrimitive<const D: usize> = JitTensor<R, Self::IntElem, D>;
    type IntElem = <R::Compiler as Compiler>::Int;
    type BoolTensorPrimitive<const D: usize> = JitTensor<R, u32, D>;

    type DynTensorPrimitive = DynJitTensor<R::Server, R::Channel, R::Device>;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        format!("jit<{}>", R::name())
    }

    fn seed(seed: u64) {
        let rng = StdRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }

    fn sync(device: &Self::Device) {
        let client = R::client(device);
        client.sync();
    }

    fn dyn_from_data(data: DynData<Self::FullPrecisionElem, Self::IntElem>, device: &Self::Device) -> Self::DynTensorPrimitive {
        match data {
            DynData::Float(dyn_rank_data) => DynJitTensor::from_dyn_rank_data::<R, Self::FloatElem>(dyn_rank_data.convert(), device),
            DynData::Int(dyn_rank_data) => DynJitTensor::from_dyn_rank_data::<R, Self::IntElem>(dyn_rank_data, device),
            DynData::Bool(dyn_rank_data) => {
                DynJitTensor::from_dyn_rank_data::<R, u32>(DynRankData::new(
                    dyn_rank_data.value
                        .into_iter()
                        .map(|boolean| boolean as u32)
                        .collect(),
                    dyn_rank_data.shape
                ), device)
            },
        }
    }

    fn dyn_into_data(dyn_tensor: Self::DynTensorPrimitive) -> DynData<Self::FullPrecisionElem, Self::IntElem> {
        match dyn_tensor.elem {
            Elem::Float => DynData::Float(dyn_tensor.into_dyn_rank_data().read()),
            Elem::Int => DynData::Int(dyn_tensor.into_dyn_rank_data().read()),
            Elem::UInt => DynData::Int(dyn_tensor.into_dyn_rank_data().read()),
            Elem::Bool => {
                let mut dyn_rank_data = dyn_tensor.into_dyn_rank_data::<u32>().read();

                DynData::Bool(
                    DynRankData::new(
                        dyn_rank_data.value
                            .into_iter()
                            .map(|boolean| boolean != 0)
                            .collect(),
                        dyn_rank_data.shape
                    )
                )
            },
        }
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
