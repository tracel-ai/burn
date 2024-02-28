use core::fmt::Debug;

use crate::{backend::Backend, BasicOps, Device, DynData, Tensor};

#[derive(new, Clone, Debug)]
pub struct DynTensor<B>
where
    B: Backend,
{
    pub(crate) primitive: B::DynTensorPrimitive,
}

impl<B> DynTensor<B>
where
    B: Backend,
{
    pub fn from_data(data: DynData, device: &Device<B>) -> Self {
        Self {
            primitive: B::dyn_from_data(data, device),
        }
    }

    pub fn into_data(self) -> DynData {
        B::dyn_into_data(self.primitive)
    }

    pub fn to_data(&self) -> DynData {
        self.clone().into_data()
    }

    pub fn from_primitive(primitive: B::DynTensorPrimitive) -> Self {
        Self { primitive }
    }

    pub fn into_primitive(self) -> B::DynTensorPrimitive {
        self.primitive
    }

    pub fn as_backend<BOut>(self) -> DynTensor<BOut>
    where
        BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>,
    {
        DynTensor {
            primitive: self.primitive,
        }
    }
}

impl<BIn, BOut, const D: usize, K> From<DynTensor<BIn>> for Tensor<BOut, D, K>
where
    BIn: Backend,
    BOut: Backend<DynTensorPrimitive = BIn::DynTensorPrimitive>,
    K: BasicOps<BOut>,
{
    fn from(value: DynTensor<BIn>) -> Self {
        Tensor::from_primitive(K::from_dyn(value.primitive))
    }
}

impl<BIn, BOut, const D: usize, K> From<Tensor<BIn, D, K>> for DynTensor<BOut>
where
    BIn: Backend,
    BOut: Backend<DynTensorPrimitive = BIn::DynTensorPrimitive>,
    K: BasicOps<BIn>,
{
    fn from(value: Tensor<BIn, D, K>) -> Self {
        Self::from_primitive(K::into_dyn(value.primitive).read())
    }
}
