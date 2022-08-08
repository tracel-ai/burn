use crate::tensor::backend::Backend;
use crate::tensor::ops::*;
use crate::tensor::{Data, Distribution, Shape};

#[derive(Debug, Clone)]
pub struct Tensor<const D: usize, B: Backend> {
    pub(crate) value: B::TensorPrimitive<D>,
}

pub struct BoolTensor<const D: usize, B: Backend> {
    pub(crate) value: B::BoolTensorPrimitive<D>,
}

impl<const D: usize, B> BoolTensor<D, B>
where
    B: Backend,
{
    pub fn new(tensor: B::BoolTensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    pub fn shape(&self) -> &Shape<D> {
        self.value.shape()
    }

    pub fn into_data(self) -> Data<bool, D> {
        self.value.into_data()
    }

    pub fn to_data(&self) -> Data<bool, D> {
        self.value.to_data()
    }

    pub fn from_data(data: Data<bool, D>) -> Self {
        let value = B::from_data_bool(data, B::Device::default());
        Self::new(value)
    }
}

impl<const D: usize, B> Tensor<D, B>
where
    B: Backend,
{
    pub fn new(tensor: B::TensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    pub fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, B> {
        Tensor::new(self.value.reshape(shape))
    }

    pub fn to_device(&self, device: B::Device) -> Self {
        Self::new(self.value.to_device(device))
    }

    pub fn device(&self) -> B::Device {
        self.value.device()
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

    pub fn zeros_like(&self) -> Self {
        Tensor::new(B::zeros(self.shape().clone(), self.value.device()))
    }

    pub fn ones_like(&self) -> Self {
        Tensor::new(B::ones(self.shape().clone(), self.value.device()))
    }

    pub fn random_like(&self, distribution: Distribution<B::Elem>) -> Self {
        Tensor::new(B::random(
            self.shape().clone(),
            distribution,
            self.value.device(),
        ))
    }

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

    pub fn random(shape: Shape<D>, distribution: Distribution<B::Elem>) -> Self {
        let tensor = B::random(shape, distribution, B::Device::default());
        Self::new(tensor)
    }

    pub fn greater(&self, other: &Self) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.greater(&other.value))
    }

    pub fn greater_equal(&self, other: &Self) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.greater_equal(&other.value))
    }

    pub fn greater_scalar(&self, other: &B::Elem) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.greater_scalar(other))
    }

    pub fn greater_equal_scalar(&self, other: &B::Elem) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.greater_equal_scalar(other))
    }

    pub fn lower(&self, other: &Self) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.lower(&other.value))
    }

    pub fn lower_equal(&self, other: &Self) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.lower_equal(&other.value))
    }

    pub fn lower_scalar(&self, other: &B::Elem) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.lower_scalar(other))
    }

    pub fn lower_equal_scalar(&self, other: &B::Elem) -> BoolTensor<D, B> {
        BoolTensor::new(self.value.lower_equal_scalar(other))
    }

    pub fn zeros(shape: Shape<D>) -> Self {
        let tensor = B::zeros(shape, B::Device::default());
        Self::new(tensor)
    }

    pub fn ones(shape: Shape<D>) -> Self {
        let tensor = B::ones(shape, B::Device::default());
        Self::new(tensor)
    }

    pub fn from_data(data: Data<B::Elem, D>) -> Self {
        let tensor = B::from_data(data, B::Device::default());
        Tensor::new(tensor)
    }

    pub fn from_data_device(data: Data<B::Elem, D>, device: B::Device) -> Self {
        let tensor = B::from_data(data, device);
        Tensor::new(tensor)
    }

    pub fn index<const D2: usize>(&self, indexes: [std::ops::Range<usize>; D2]) -> Self {
        Self::new(self.value.index(indexes))
    }

    pub fn index_assign<const D2: usize>(
        &self,
        indexes: [std::ops::Range<usize>; D2],
        values: &Self,
    ) -> Self {
        Self::new(self.value.index_assign(indexes, &values.value))
    }

    pub fn mask_fill(&self, mask: &BoolTensor<D, B>, value: B::Elem) -> Self {
        Self::new(self.value.mask_fill(&mask.value, value))
    }

    pub fn unsqueeze<const D2: usize>(&self) -> Tensor<D2, B> {
        if D2 < D {
            panic!(
                "Can't unsqueeze smaller tensor, got dim {}, expected > {}",
                D2, D
            )
        }

        let mut dims = [1; D2];
        let num_ones = D2 - D;
        let shape = self.shape();

        for i in 0..D {
            dims[i + num_ones] = shape.dims[i];
        }

        let shape = Shape::new(dims);
        self.reshape(shape)
    }
}
