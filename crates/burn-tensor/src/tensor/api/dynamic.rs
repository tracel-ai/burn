use core::fmt::Debug;

use crate::{backend::Backend, BasicOps, Device, DynData, Element, Tensor};

// TODO: Replace this once bound aliases become a thing
/// Marker trait to specify a backend that uses a specific type as its dynamic tensor primitive.
pub trait DynPrimBackend<P>: Backend<DynTensorPrimitive = P> {}

/// Marker trait to specify that a backend is compatible with another backend's dynamic tensor primitive.
pub trait DynCompatBackend<B: Backend>: Backend<DynTensorPrimitive = B::DynTensorPrimitive> {}

impl<P, B: Backend<DynTensorPrimitive = P>> DynPrimBackend<P> for B {}

impl<B: Backend, CompatB: Backend<DynTensorPrimitive = B::DynTensorPrimitive>> DynCompatBackend<B> for CompatB {}

#[derive(Clone, Debug)]
/// A dynamic tensor using a given underlying dynamic tensor primitive.
///
/// This type is mainly designed for use in [TensorContainer], to store tensors of arbitrary rank and element type in the same container.
pub struct DynTensor<P>
{
    pub(crate) primitive: P,
}

impl<P> DynTensor<P> {
    /// Create a dynamic tensor from [DynData], on a given device.
    pub fn from_dyn_data<FElem: Element, IElem: Element, B: Backend<DynTensorPrimitive = P>>(data: DynData<FElem, IElem>, device: &Device<B>) -> Self {
        Self {
            primitive: B::dyn_from_data(data.convert(), device),
        }
    }

    /// Convert a dynamic tensor into [DynData].
    pub fn into_dyn_data<B: Backend<DynTensorPrimitive = P>>(self) -> DynData<B::FullPrecisionElem, B::IntElem> {
        B::dyn_into_data(self.primitive)
    }

    /// Wraps a primitive in a [DynTensor].
    pub fn from_primitive(primitive: P) -> Self {
        Self { primitive }
    }

    /// Returns the underlying primitive used in this [DynTensor].
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
        Self::from_primitive(K::into_dyn(value.into_primitive()))
    }
}