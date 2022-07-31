use super::backend::autodiff::ADTensor;
use super::backend::ADBackend;
use super::backend::Backend;
use crate::graph::grad::Gradients;
use crate::tensor::ops::*;
use crate::tensor::Element;
use crate::tensor::{Data, Distribution, Shape};
use rand::distributions::Standard;

#[derive(Debug, Clone)]
pub struct Tensor<const D: usize, B: Backend> {
    pub value: B::Tensor<D>,
}

impl<const D: usize, B: ADBackend> Tensor<D, B> {
    pub fn backward(&self) -> Gradients {
        B::backward::<D>(&self.value)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<D, B::InnerBackend>> {
        B::grad(&self.value, grads).map(|value| Tensor::new(value))
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use crate::tensor::backend::autodiff::ADBackendNdArray;
    use crate::tensor::backend::ndarray::NdArrayBackend;

    impl<E: Element, const D: usize> Tensor<D, NdArrayBackend<E>>
    where
        Standard: rand::distributions::Distribution<E>,
    {
        pub fn with_grad(self) -> Tensor<D, ADBackendNdArray<E>> {
            let tensor = ADTensor::from_tensor(self.value);
            Tensor::new(tensor)
        }
    }
}

#[cfg(feature = "tch")]
mod tch {
    use super::*;
    use crate::tensor::backend::autodiff::ADBackendTch;
    use crate::tensor::backend::tch::TchBackend;

    impl<E: Element, const D: usize> Tensor<D, TchBackend<E>>
    where
        Standard: rand::distributions::Distribution<E>,
    {
        pub fn with_grad(self) -> Tensor<D, ADBackendTch<E>> {
            let tensor = ADTensor::from_tensor(self.value);
            Tensor::new(tensor)
        }
    }
}

impl<const D: usize, B> std::ops::Add<Self> for Tensor<D, B>
where
    B: Backend,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        Self::new(value)
    }
}

impl<const D: usize, B> Tensor<D, B>
where
    B: Backend,
{
    pub fn new(tensor: B::Tensor<D>) -> Self {
        Self { value: tensor }
    }

    pub fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, B> {
        Tensor::new(self.value.reshape(shape))
    }

    pub fn shape(&self) -> &Shape<D> {
        self.value.shape()
    }

    pub fn into_data(self) -> Data<B::Elem, D> {
        self.value.into_data()
    }

    pub fn to_data(&self) -> Data<B::Elem, D> {
        self.value.to_data()
    }

    pub fn new_like_empty(&self) -> Self {
        Self::new(self.value.new_like_empty())
    }

    pub fn new_like_random(&self, distribution: Distribution<B::Elem>) -> Self {
        Self::new(self.value.new_like_random(distribution))
    }

    pub fn new_like_data(&self, data: Data<B::Elem, D>) -> Self {
        Self::new(self.value.new_like_data(data))
    }

    pub fn new_like_zeros(&self) -> Self {
        Self::new(self.value.new_like_zeros())
    }

    pub fn new_like_ones(&self) -> Self {
        Self::new(self.value.new_like_ones())
    }

    // pub fn new_fork_empty<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, B> {
    //     Tensor::new(self.value.new_fork_empty(shape))
    // }

    // pub fn new_fork_random<const D2: usize>(
    //     &self,
    //     shape: Shape<D2>,
    //     distribution: Distribution<B::Elem>,
    // ) -> Tensor<D2, B> {
    //     Tensor::new(self.value.new_fork_random(shape, distribution))
    // }

    // pub fn new_fork_data<const D2: usize>(&self, data: Data<B::Elem, D2>) -> Tensor<D2, B> {
    //     Tensor::new(self.value.new_fork_data(data))
    // }

    // pub fn new_fork_zeros<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, B> {
    //     Tensor::new(self.value.new_fork_zeros(shape))
    // }

    // pub fn new_fork_ones<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, B> {
    //     Tensor::new(self.value.new_fork_ones(shape))
    // }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.value.add(&other.value))
    }

    pub fn add_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.add_scalar(&other))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.value.sub(&other.value))
    }

    pub fn sub_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.sub_scalar(&other))
    }

    pub fn transpose(&self) -> Self {
        Self::new(self.value.transpose())
    }

    pub fn matmul(&self, other: &Self) -> Self {
        Self::new(self.value.matmul(&other.value))
    }

    pub fn neg(&self) -> Self {
        Self::new(self.value.neg())
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::new(self.value.mul(&other.value))
    }

    pub fn mul_scalar(&self, other: &B::Elem) -> Self {
        Self::new(self.value.mul_scalar(&other))
    }

    pub fn from_data(data: Data<B::Elem, D>) -> Self {
        let tensor = B::from_data(data, B::Device::default());
        Tensor::new(tensor)
    }

    pub fn from_data_device(data: Data<B::Elem, D>, device: B::Device) -> Self {
        let tensor = B::from_data(data, device);
        Tensor::new(tensor)
    }
    // pub fn index<const D2: usize>(&self, indexes: [std::ops::Range<usize>; D2]) -> Self
    // where
    //     TensorOps<D, B>: TensorOpsIndex<B::Elem, D, D2>,
    // {
    //     Self::new(self.value.index(indexes))
    // }
}
