use crate::element::FloatNdArrayElement;
use crate::NdArrayTensor;
use alloc::string::String;
use burn_common::stub::Mutex;
use burn_tensor::backend::Backend;
use burn_tensor::{DynData, DynRankData};
use core::marker::PhantomData;
use ndarray::{ArcArray, IxDyn};
use rand::{rngs::StdRng, SeedableRng};

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdArrayDevice {
    /// The CPU device.
    Cpu,
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

#[derive(Debug, Clone)]
/// An n-dimensional array with a dynamic rank, and a dynamic specific element type.
pub enum DynNdArray<F, I> {
    /// An n-dimensional array storing floats.
    Float(ArcArray<F, IxDyn>),
    /// An n-dimensional array storing integers.
    Int(ArcArray<I, IxDyn>),
    /// An n-dimensional array storing booleans.
    Bool(ArcArray<bool, IxDyn>),
}

impl<E: FloatNdArrayElement> Backend for NdArray<E> {
    type Device = NdArrayDevice;
    type FullPrecisionBackend = NdArray<f32>;
    type FullPrecisionElem = f32;

    type FloatTensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type FloatElem = E;

    type IntTensorPrimitive<const D: usize> = NdArrayTensor<i64, D>;
    type IntElem = i64;

    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;

    type DynTensorPrimitive = DynNdArray<Self::FullPrecisionElem, Self::IntElem>;

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

    fn dyn_from_data(
        data: DynData<Self::FullPrecisionElem, Self::IntElem>,
        _device: &Self::Device,
    ) -> Self::DynTensorPrimitive {
        match data {
            DynData::Float(data) => {
                DynNdArray::Float(ArcArray::from_vec(data.value).reshape(data.shape))
            }
            DynData::Int(data) => {
                DynNdArray::Int(ArcArray::from_vec(data.value).reshape(data.shape))
            }
            DynData::Bool(data) => {
                DynNdArray::Bool(ArcArray::from_vec(data.value).reshape(data.shape))
            }
        }
    }

    fn dyn_into_data(
        dyn_tensor: Self::DynTensorPrimitive,
    ) -> DynData<Self::FullPrecisionElem, Self::IntElem> {
        match dyn_tensor {
            DynNdArray::Float(arc_array) => DynData::Float(DynRankData::new(
                arc_array.clone().into_iter().collect(),
                arc_array.shape().to_vec(),
            )),
            DynNdArray::Int(arc_array) => DynData::Int(DynRankData::new(
                arc_array.clone().into_iter().collect(),
                arc_array.shape().to_vec(),
            )),
            DynNdArray::Bool(arc_array) => DynData::Bool(DynRankData::new(
                arc_array.clone().into_iter().collect(),
                arc_array.shape().to_vec(),
            )),
        }
    }
}
