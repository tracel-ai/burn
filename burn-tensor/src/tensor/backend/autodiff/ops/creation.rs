use crate::tensor::{
    backend::autodiff::ADTensor, ops::*, Backend, Data, Distribution, Element, Shape, Tensor,
    TensorTrait, TensorType,
};
use rand::distributions::Standard;

impl<E, const D: usize, B> TensorOpsCreation<E, D, B> for ADTensor<E, D, Tensor<D, B>>
where
    E: Element,
    B: Backend<E = E> + TensorType<D, B>,
    Standard: rand::distributions::Distribution<E>,
{
    fn grad(&self) -> Tensor<D, B> {
        panic!();
    }
}

impl<P, const D: usize, T> TensorCreationLike<P, D> for ADTensor<P, D, T>
where
    T: TensorTrait<P, D> + TensorCreationLike<P, D>,
    P: Element,
    Standard: rand::distributions::Distribution<P>,
{
    fn new_like_empty(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_empty())
    }

    fn new_like_random(&self, distribution: Distribution<P>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_random(distribution))
    }

    fn new_like_data(&self, data: Data<P, D>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_data(data))
    }

    fn new_like_zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_zeros())
    }

    fn new_like_ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_ones())
    }
}

impl<P, const D: usize, const D2: usize, T, T2> TensorCreationFork<P, D, D2> for ADTensor<P, D, T>
where
    T: TensorTrait<P, D> + TensorCreationFork<P, D, D2, Output = T2>,
    T2: TensorTrait<P, D2>,
    P: Element,
    Standard: rand::distributions::Distribution<P>,
{
    type Output = ADTensor<P, D2, T2>;

    fn new_fork_empty(&self, shape: Shape<D2>) -> ADTensor<P, D2, T2> {
        ADTensor::from_tensor(self.tensor().new_fork_empty(shape))
    }

    fn new_fork_random(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<P>,
    ) -> ADTensor<P, D2, T2> {
        ADTensor::from_tensor(self.tensor().new_fork_random(shape, distribution))
    }

    fn new_fork_data(&self, data: Data<P, D2>) -> ADTensor<P, D2, T2> {
        ADTensor::from_tensor(self.tensor().new_fork_data(data))
    }

    fn new_fork_zeros(&self, shape: Shape<D2>) -> ADTensor<P, D2, T2> {
        ADTensor::from_tensor(self.tensor().new_fork_zeros(shape))
    }

    fn new_fork_ones(&self, shape: Shape<D2>) -> ADTensor<P, D2, T2> {
        ADTensor::from_tensor(self.tensor().new_fork_ones(shape))
    }
}

impl<T, P, const D: usize> Zeros<Self> for ADTensor<P, D, T>
where
    P: Element,
    T: TensorTrait<P, D>,
{
    fn zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().zeros())
    }
}

impl<T, P, const D: usize> Ones<Self> for ADTensor<P, D, T>
where
    P: Element,
    T: TensorTrait<P, D>,
{
    fn ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().ones())
    }
}
