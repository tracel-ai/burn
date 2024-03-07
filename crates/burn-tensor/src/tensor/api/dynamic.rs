use core::fmt::Debug;

use crate::{backend::Backend, BasicOps, Device, DynData, Tensor};

// TODO: Replace this once bound aliases become a thing
pub trait DynPrimBackend<P>: Backend<DynTensorPrimitive = P> {}

pub trait DynCompatBackend<B: Backend>: Backend<DynTensorPrimitive = B::DynTensorPrimitive> {}

impl<P, B: Backend<DynTensorPrimitive = P>> DynPrimBackend<P> for B {}

impl<B: Backend, CompatB: Backend<DynTensorPrimitive = B::DynTensorPrimitive>> DynCompatBackend<B> for CompatB {}

#[derive(Clone, Debug)]
pub struct DynTensor<P>
{
    pub(crate) primitive: P,
}

impl<P> DynTensor<P> {
    pub fn from_data<B: Backend<DynTensorPrimitive = P>>(data: DynData<B::FullPrecisionElem, B::IntElem>, device: &Device<B>) -> Self {
        Self {
            primitive: B::dyn_from_data(data, device),
        }
    }

    pub fn into_data<B: Backend<DynTensorPrimitive = P>>(self) -> DynData<B::FullPrecisionElem, B::IntElem> {
        B::dyn_into_data(self.primitive)
    }

    pub fn from_primitive(primitive: P) -> Self {
        Self { primitive }
    }

    pub fn into_primitive(self) -> P {
        self.primitive
    }
}


impl<P, B, const D: usize, K> From<DynTensor<P>> for Tensor<B, D, K>
where
    B: DynPrimBackend<P>,
    K: BasicOps<B>,
{
    fn from(value: DynTensor<P>) -> Self {
        Tensor::from_primitive(K::from_dyn(value.into_primitive()))
    }
}

impl<P, B, const D: usize, K> From<Tensor<B, D, K>> for DynTensor<P>
    where
        B: DynPrimBackend<P>,
        K: BasicOps<B>,
{
    fn from(value: Tensor<B, D, K>) -> Self {
        Self::from_primitive(K::into_dyn(value.into_primitive()).read())
    }
}