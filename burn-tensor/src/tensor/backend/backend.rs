use crate::graph::grad::Gradients;
use crate::tensor::ops::{TensorCreationFork, TensorOpsIndex, TensorOpsReshape};
use crate::tensor::Data;
use crate::tensor::{ops::TensorCreationLike, Element, TensorTrait};

pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device: Copy + Clone + Default;
    type Elem: Element;
    type Tensor<const D: usize>: TensorTrait<Self::Elem, D>
        + TensorCreationLike<Self::Elem, D>
        + TensorCreationFork<Self, D>
        + TensorOpsReshape<Self, D>
        + TensorOpsIndex<Self::Elem, D>
        + 'static;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::Tensor<D>;
    fn ad_enabled() -> bool;
    fn name() -> String;
}

pub type ADBackendTensor<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::Tensor<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend;

    fn backward<const D: usize>(tensor: &Self::Tensor<D>) -> Gradients;
    fn grad<const D: usize>(
        tensor: &Self::Tensor<D>,
        grads: &Gradients,
    ) -> Option<ADBackendTensor<D, Self>>;
}
