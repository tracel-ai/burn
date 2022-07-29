use super::{
    ops::{TensorCreationFork, TensorCreationLike, TensorOpsIndex, TensorOpsReshape},
    Data, Element, TensorTrait,
};

pub type Tensor<const D: usize, B> = <B as TensorType<D, B>>::T;

pub trait Backend:
    Sized
    + Default
    + Send
    + Sync
    + std::fmt::Debug
    + TensorType<1, Self>
    + TensorType<2, Self>
    + TensorType<3, Self>
    + TensorType<4, Self>
    + TensorType<5, Self>
    + TensorType<6, Self>
{
    type E: Element;
    type Device: Default;

    fn from_data<const D: usize>(
        data: Data<Self::E, D>,
        device: Self::Device,
    ) -> <Self as TensorType<D, Self>>::T
    where
        Self: TensorType<D, Self>;
}

pub trait TensorType<const D: usize, B: Backend> {
    type T: TensorTrait<B::E, D>
        + TensorCreationLike<B::E, D>
        + TensorCreationFork<B::E, D, 1, Output = Tensor<1, B>>
        + TensorCreationFork<B::E, D, 2, Output = Tensor<2, B>>
        + TensorCreationFork<B::E, D, 3, Output = Tensor<3, B>>
        + TensorCreationFork<B::E, D, 4, Output = Tensor<4, B>>
        + TensorCreationFork<B::E, D, 5, Output = Tensor<5, B>>
        + TensorCreationFork<B::E, D, 6, Output = Tensor<6, B>>
        + TensorOpsIndex<B::E, D, 1>
        + TensorOpsIndex<B::E, D, 2>
        + TensorOpsIndex<B::E, D, 3>
        + TensorOpsIndex<B::E, D, 4>
        + TensorOpsIndex<B::E, D, 5>
        + TensorOpsIndex<B::E, D, 6>
        + TensorOpsReshape<B::E, D, B>;

    fn from_data(data: Data<B::E, D>, device: B::Device) -> Self::T;
}
