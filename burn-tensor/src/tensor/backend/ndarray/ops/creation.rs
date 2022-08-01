use crate::tensor::{
    backend::ndarray::{NdArrayBackend, NdArrayTensor},
    ops::*,
    Data, Distribution, Element, Shape,
};
use rand::distributions::Standard;

impl<P, const D: usize> TensorCreationLike<P, D> for NdArrayTensor<P, D>
where
    P: Element,
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
        self.zeros()
    }

    fn new_like_ones(&self) -> Self {
        self.ones()
    }
}

impl<P: Element, const D: usize> TensorCreationFork<NdArrayBackend<P>, D> for NdArrayTensor<P, D>
where
    Standard: rand::distributions::Distribution<P>,
{
    fn new_fork_empty<const D2: usize>(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        self.new_fork_zeros(shape)
    }

    fn new_fork_random<const D2: usize>(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<P>,
    ) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::random(shape, distribution);
        NdArrayTensor::from_data(data)
    }

    fn new_fork_data<const D2: usize>(&self, data: Data<P, D2>) -> NdArrayTensor<P, D2> {
        NdArrayTensor::from_data(data)
    }

    fn new_fork_zeros<const D2: usize>(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::zeros(shape);
        NdArrayTensor::from_data(data)
    }

    fn new_fork_ones<const D2: usize>(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
        let data = Data::<P, D2>::ones(shape);
        NdArrayTensor::from_data(data)
    }
}

impl<P, const D: usize> Zeros<NdArrayTensor<P, D>> for NdArrayTensor<P, D>
where
    P: Default + Clone + Zeros<P> + std::fmt::Debug,
{
    fn zeros(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::zeros(self.shape.clone());
        Self::from_data(data)
    }
}

impl<P, const D: usize> Ones<NdArrayTensor<P, D>> for NdArrayTensor<P, D>
where
    P: Default + Clone + Ones<P> + std::fmt::Debug,
{
    fn ones(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::ones(self.shape.clone());
        Self::from_data(data)
    }
}
