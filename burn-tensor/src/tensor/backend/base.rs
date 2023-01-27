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
    + std::fmt::Debug
    + 'static
{
    type Device: Clone + Default + std::fmt::Debug + Send + Sync;
    type Elem: Element;
    type FullPrecisionElem: Element;
    type FullPrecisionBackend: Backend<Elem = Self::FullPrecisionElem, Device = Self::Device>;
    type IntegerBackend: Backend<Elem = i64, Device = Self::Device>;
    type TensorPrimitive<const D: usize>: std::ops::Add<Self::TensorPrimitive<D>, Output = Self::TensorPrimitive<D>>
        + Zeros
        + Ones
        + Clone
        + Send
        + Sync
        + Send
        + Sync
        + 'static
        + std::fmt::Debug;

    type BoolTensorPrimitive<const D: usize>: Clone
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + From<<Self::IntegerBackend as Backend>::BoolTensorPrimitive<D>>;

    fn ad_enabled() -> bool;
    fn name() -> String;
    fn seed(seed: u64);
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, Elem = Self::Elem>;
    type Gradients: Send + Sync;

    fn backward<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> Self::Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Self::Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
