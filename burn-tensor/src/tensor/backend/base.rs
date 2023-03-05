use alloc::string::String;

use crate::ops::*;
use crate::tensor::Element;

pub trait Backend:
    TensorOps<Self>
    + ModuleOps<Self>
    + Clone
    + Sized
    + Default
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
{
    type Device: Clone + Default + core::fmt::Debug + Send + Sync;
    type FloatElem: Element;
    type IntElem: Element;
    type FullPrecisionElem: Element;
    type FullPrecisionBackend: Backend<FloatElem = Self::FullPrecisionElem, Device = Self::Device>;
    type IntegerBackend: Backend<FloatElem = i64, Device = Self::Device>;
    type TensorPrimitive<const D: usize>: Clone
        + Send
        + Sync
        + Send
        + Sync
        + 'static
        + core::fmt::Debug;

    type BoolTensorPrimitive<const D: usize>: Clone
        + Send
        + Sync
        + 'static
        + core::fmt::Debug
        + From<<Self::IntegerBackend as Backend>::BoolTensorPrimitive<D>>;

    fn ad_enabled() -> bool;
    fn name() -> String;
    fn seed(seed: u64);
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem>;
    type Gradients: Send + Sync;

    fn backward<const D: usize>(tensor: Self::TensorPrimitive<D>) -> Self::Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Self::Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn grad_remove<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &mut Self::Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
