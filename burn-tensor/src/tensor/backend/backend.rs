use crate::graph::grad::Gradients;
use crate::tensor::{ops::TensorCreationLike, Element, TensorTrait};

pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device;
    type Elem: Element;
    type Tensor<const D: usize>: TensorTrait<Self::Elem, D>
        + TensorCreationLike<Self::Elem, D>
        + 'static;
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
