use crate::tensor::{
    backend::tch::{TchShape, TchTensor},
    ops::*,
    Data, Distribution, Shape,
};
use rand::distributions::{uniform::SampleUniform, Standard};

impl<P, const D: usize> TensorCreationLike<P, D> for TchTensor<P, D>
where
    P: tch::kind::Element + std::fmt::Debug + SampleUniform + Default,
    Standard: rand::distributions::Distribution<P>,
{
    fn new_like_empty(&self) -> Self {
        let tensor = self.tensor.empty_like();
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            kind,
            tensor,
            shape,
        }
    }

    fn new_like_random(&self, distribution: Distribution<P>) -> Self {
        let device = self.tensor.device();
        let data = Data::<P, D>::random(self.shape.clone(), distribution);

        Self::from_data(data, device)
    }

    fn new_like_data(&self, data: Data<P, D>) -> Self {
        let device = self.tensor.device();
        Self::from_data(data, device)
    }

    fn new_like_zeros(&self) -> Self {
        let tensor = self.tensor.zeros_like();
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            kind,
            tensor,
            shape,
        }
    }

    fn new_like_ones(&self) -> Self {
        let tensor = self.tensor.ones_like();
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self {
            kind,
            tensor,
            shape,
        }
    }
}

impl<P, const D: usize, const D2: usize> TensorCreationFork<P, D, D2, TchTensor<P, D2>>
    for TchTensor<P, D>
where
    P: tch::kind::Element + std::fmt::Debug + SampleUniform + Default + Copy,
    Standard: rand::distributions::Distribution<P>,
{
    fn new_fork_empty(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
        let device = self.tensor.device();
        let kind = self.kind.clone();

        let tch_shape = TchShape::from(shape.clone());
        let tensor = tch::Tensor::empty(&tch_shape.dims, (kind.kind(), device));

        TchTensor {
            kind,
            tensor,
            shape,
        }
    }

    fn new_fork_random(&self, shape: Shape<D2>, distribution: Distribution<P>) -> TchTensor<P, D2> {
        let device = self.tensor.device();
        let data = Data::<P, D2>::random(shape, distribution);

        TchTensor::from_data(data, device)
    }

    fn new_fork_data(&self, data: Data<P, D2>) -> TchTensor<P, D2> {
        let device = self.tensor.device();
        TchTensor::from_data(data, device)
    }

    fn new_fork_zeros(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
        let device = self.tensor.device();
        let kind = self.kind.clone();

        let tch_shape = TchShape::from(shape.clone());
        let tensor = tch::Tensor::zeros(&tch_shape.dims, (kind.kind(), device));

        TchTensor {
            kind,
            tensor,
            shape,
        }
    }

    fn new_fork_ones(&self, shape: Shape<D2>) -> TchTensor<P, D2> {
        let device = self.tensor.device();
        let kind = self.kind.clone();

        let tch_shape = TchShape::from(shape.clone());
        let tensor = tch::Tensor::ones(&tch_shape.dims, (kind.kind(), device));

        TchTensor {
            kind,
            tensor,
            shape,
        }
    }
}
