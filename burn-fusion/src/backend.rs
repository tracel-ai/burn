use burn_tensor::backend::Backend;
use core::marker::PhantomData;

use crate::FusionTensor;

#[derive(Clone, Debug, Default)]
pub struct FusionBackend<B> {
    _backend: PhantomData<B>,
}

impl<B: Backend> Backend for FusionBackend<B> {
    type Device = B::Device;

    type FullPrecisionBackend = FusionBackend<B::FullPrecisionBackend>;

    type FullPrecisionElem = B::FullPrecisionElem;

    type TensorPrimitive<const D: usize> = FusionTensor<B, B::FloatElem>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<B, B::IntElem>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<B, bool>;

    fn name() -> String {
        todo!()
    }

    fn seed(seed: u64) {
        todo!()
    }
}

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type FullPrecisionBackend<B> = <B as Backend>::FullPrecisionBackend;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;

pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;
