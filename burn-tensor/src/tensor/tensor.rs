use super::backend::autodiff::ADTensor;
use super::ops::*;
use super::{ops::TensorOpsReshape, Data, Element, Shape, TensorTrait};

type E<B> = <B as Backend>::E;
pub type Tensor<const D: usize, B> = <B as TensorType<E<B>, D, B>>::T;

pub trait Backend:
    Sized
    + TensorType<Self::E, 1, Self>
    + TensorType<Self::E, 2, Self>
    + TensorType<Self::E, 3, Self>
    + TensorType<Self::E, 4, Self>
    + TensorType<Self::E, 5, Self>
    + TensorType<Self::E, 6, Self>
{
    type E: Element;

    fn from_data<const D: usize>(
        data: Data<Self::E, D>,
    ) -> <Self as TensorType<Self::E, D, Self>>::T
    where
        Self: TensorType<Self::E, D, Self>;
}

pub trait TensorType<E: Element, const D: usize, B: Backend> {
    type T: TensorTrait<E, D>
        + TensorCreationLike<E, D>
        + TensorCreationFork<E, D, 2, Output = Tensor<1, B>>
        + TensorCreationFork<E, D, 2, Output = Tensor<2, B>>
        + TensorCreationFork<E, D, 3, Output = Tensor<3, B>>
        + TensorCreationFork<E, D, 4, Output = Tensor<4, B>>
        + TensorCreationFork<E, D, 5, Output = Tensor<5, B>>
        + TensorCreationFork<E, D, 6, Output = Tensor<6, B>>
        + TensorOpsIndex<E, D, 1>
        + TensorOpsIndex<E, D, 2>
        + TensorOpsIndex<E, D, 3>
        + TensorOpsIndex<E, D, 4>
        + TensorOpsIndex<E, D, 5>
        + TensorOpsIndex<E, D, 6>
        + TensorOpsReshape<E, D, 1, Output = Tensor<1, B>>
        + TensorOpsReshape<E, D, 2, Output = Tensor<2, B>>
        + TensorOpsReshape<E, D, 3, Output = Tensor<3, B>>
        + TensorOpsReshape<E, D, 4, Output = Tensor<4, B>>
        + TensorOpsReshape<E, D, 5, Output = Tensor<5, B>>
        + TensorOpsReshape<E, D, 6, Output = Tensor<6, B>>;

    fn from_data(data: Data<E, D>) -> Self::T;
}

fn allo<B: Backend>(tensor: &Tensor<1, B>) {
    let t1 = tensor.reshape(Shape::new([1, 1, 1, 1]));
    let t3 = t1.reshape(Shape::new([1, 1, 1, 1]));
    let t4 = t1.add(&t3);
    let forked = t4.new_fork_empty(Shape::new([2, 2])); // Very complex type
    let ad = ADTensor::from_tensor(forked);
}
