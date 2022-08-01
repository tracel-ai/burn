use crate::graph::grad::Gradients;
use crate::ops::TensorOpsDevice;
use crate::tensor::ops::{TensorCreationFork, TensorOpsIndex, TensorOpsReshape};
use crate::tensor::{ops::TensorCreationLike, Element, TensorTrait};
use crate::tensor::{Data, Distribution, Shape};

pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device: Copy + Clone + Default + std::fmt::Debug + Send + Sync;
    type Elem: Element;
    type TensorPrimitive<const D: usize>: TensorTrait<Self::Elem, D>
        + TensorCreationLike<Self::Elem, D>
        + TensorCreationFork<Self, D>
        + TensorOpsReshape<Self, D>
        + TensorOpsDevice<Self, D>
        + TensorOpsIndex<Self::Elem, D>
        + 'static;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D>;

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D>;

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D>;

    fn ad_enabled() -> bool;
    fn name() -> String;
}

pub type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend;

    fn backward<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
