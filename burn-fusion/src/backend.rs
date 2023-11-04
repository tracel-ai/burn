use crate::client::mutex::MutexFusionClient;
use crate::graph::GreedyGraphExecution;
use crate::{graph::FusedBackend, FusionTensor};
use burn_tensor::backend::Backend;
use core::marker::PhantomData;

type Client<B> = MutexFusionClient<B, GreedyGraphExecution>;

#[derive(Clone, Debug, Default)]
pub struct FusionBackend<B> {
    _backend: PhantomData<B>,
}

impl<B: FusedBackend> Backend for FusionBackend<B> {
    type Device = B::Device;

    type FullPrecisionBackend = FusionBackend<B::FullPrecisionFusedBackend>;

    type FullPrecisionElem = B::FullPrecisionElem;

    type TensorPrimitive<const D: usize> = FusionTensor<Client<B>>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = FusionTensor<Client<B>>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = FusionTensor<Client<B>>;

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
