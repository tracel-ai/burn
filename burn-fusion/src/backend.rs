use crate::{graph::FusedBackend, FusionTensor};
use burn_tensor::backend::Backend;
use core::marker::PhantomData;

#[derive(Clone, Debug, Default)]
pub struct FusionBackend<B> {
    _backend: PhantomData<B>,
}

impl<B: FusedBackend> Backend for FusionBackend<B> {
    type Device = B::Device;

    type FullPrecisionBackend = FusionBackend<B::FullPrecisionFusedBackend>;

    type FullPrecisionElem = B::FullPrecisionElem;

    type TensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<B::FusionClient>;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}
