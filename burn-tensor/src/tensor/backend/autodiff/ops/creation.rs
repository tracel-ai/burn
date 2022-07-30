use crate::tensor::{
    backend::{autodiff::ADTensor, Backend},
    ops::*,
    Data, Distribution, Shape,
};
use rand::distributions::Standard;

impl<B: Backend, const D: usize> TensorCreationLike<B::Elem, D> for ADTensor<D, B>
where
    Standard: rand::distributions::Distribution<B::Elem>,
{
    fn new_like_empty(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_empty())
    }

    fn new_like_random(&self, distribution: Distribution<B::Elem>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_random(distribution))
    }

    fn new_like_data(&self, data: Data<B::Elem, D>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_data(data))
    }

    fn new_like_zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_zeros())
    }

    fn new_like_ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_ones())
    }
}

impl<B: Backend, const D: usize, const D2: usize>
    TensorCreationFork<B::Elem, D, D2, ADTensor<D2, B>> for ADTensor<D, B>
where
    Standard: rand::distributions::Distribution<B::Elem>,
{
    fn new_fork_empty(&self, shape: Shape<D2>) -> ADTensor<D2, B> {
        ADTensor::from_tensor(self.tensor().new_fork_empty(shape))
    }

    fn new_fork_random(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<B::Elem>,
    ) -> ADTensor<D2, B> {
        ADTensor::from_tensor(self.tensor().new_fork_random(shape, distribution))
    }

    fn new_fork_data(&self, data: Data<B::Elem, D2>) -> ADTensor<D2, B> {
        ADTensor::from_tensor(self.tensor().new_fork_data(data))
    }

    fn new_fork_zeros(&self, shape: Shape<D2>) -> ADTensor<D2, B> {
        ADTensor::from_tensor(self.tensor().new_fork_zeros(shape))
    }

    fn new_fork_ones(&self, shape: Shape<D2>) -> ADTensor<D2, B> {
        ADTensor::from_tensor(self.tensor().new_fork_ones(shape))
    }
}

impl<B: Backend, const D: usize> Zeros<Self> for ADTensor<D, B> {
    fn zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().zeros())
    }
}

impl<B: Backend, const D: usize> Ones<Self> for ADTensor<D, B> {
    fn ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().ones())
    }
}
