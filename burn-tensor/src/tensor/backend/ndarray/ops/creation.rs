use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*, Data, Distribution, Shape};
use ndarray::{Dim, Dimension};
use rand::distributions::{uniform::SampleUniform, Standard};

impl<P, const D: usize> TensorCreationLike<P, D> for NdArrayTensor<P, D>
where
    P: std::fmt::Debug + SampleUniform + Default + Clone + Zeros<P> + Ones<P>,
    Standard: rand::distributions::Distribution<P>,
{
    fn new_like_empty(&self) -> Self {
        self.new_like_zeros()
    }

    fn new_like_random(&self, distribution: Distribution<P>) -> Self {
        let data = Data::<P, D>::random(self.shape.clone(), distribution);
        Self::from_data(data)
    }

    fn new_like_data(&self, data: Data<P, D>) -> Self {
        Self::from_data(data)
    }

    fn new_like_zeros(&self) -> Self {
        let data = Data::<P, D>::zeros(self.shape.clone());
        Self::from_data(data)
    }

    fn new_like_ones(&self) -> Self {
        let data = Data::<P, D>::ones(self.shape.clone());
        Self::from_data(data)
    }
}

impl<P, const D: usize, const D2: usize> TensorCreationFork<P, D, D2, NdArrayTensor<P, D2>>
    for NdArrayTensor<P, D>
where
    P: std::fmt::Debug + SampleUniform + Default + Clone + Zeros<P> + Ones<P>,
    Dim<[usize; D2]>: Dimension,
    Standard: rand::distributions::Distribution<P>,
{
    fn new_fork_empty(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        self.new_fork_zeros(shape)
    }

    fn new_fork_random(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<P>,
    ) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::random(shape, distribution);
        NdArrayTensor::from_data(data)
    }

    fn new_fork_data(&self, data: Data<P, D2>) -> NdArrayTensor<P, D2> {
        NdArrayTensor::from_data(data)
    }

    fn new_fork_zeros(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::zeros(shape);
        NdArrayTensor::from_data(data)
    }

    fn new_fork_ones(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::ones(shape);
        NdArrayTensor::from_data(data)
    }
}
