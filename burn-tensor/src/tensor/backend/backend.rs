use crate::tensor::{ops::TensorCreationLike, Element, TensorTrait};

pub trait Backend: Clone + Sized + Default + Send + Sync + std::fmt::Debug + 'static {
    type Device;
    type Elem: Element;
    type Tensor<const D: usize>: TensorTrait<Self::Elem, D>
        + TensorCreationLike<Self::Elem, D>
        + 'static;
}
