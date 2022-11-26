use super::Tensor;
use crate::tensor::backend::Backend;
use crate::tensor::{Data, Shape};

#[derive(Debug, Clone)]
pub struct BoolTensor<B: Backend, const D: usize> {
    pub(crate) value: B::BoolTensorPrimitive<D>,
}

impl<B, const D: usize> BoolTensor<B, D>
where
    B: Backend,
{
    pub fn new(tensor: B::BoolTensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    pub fn shape(&self) -> &Shape<D> {
        B::bool_shape(&self.value)
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// Equivalent to `tensor.shape().dims`.
    pub fn dims(&self) -> [usize; D] {
        self.shape().dims
    }

    pub fn into_data(self) -> Data<bool, D> {
        B::bool_into_data(self.value)
    }

    pub fn to_data(&self) -> Data<bool, D> {
        B::bool_to_data(&self.value)
    }

    pub fn from_data(data: Data<bool, D>) -> Self {
        let value = B::from_data_bool(data, B::Device::default());
        Self::new(value)
    }

    pub fn to_int(&self) -> Tensor<B::IntegerBackend, D> {
        let data = B::bool_to_data(&self.value);
        Tensor::from_data(data.convert())
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// # Panics
    ///
    /// If the tensor can not be reshape to the given shape.
    pub fn reshape<const D2: usize, S: Into<Shape<D2>>>(&self, shape: S) -> BoolTensor<B, D2> {
        BoolTensor::new(B::bool_reshape(&self.value, shape.into()))
    }
}
