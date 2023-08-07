use crate::{backend::Backend, Bool, Data, Int, Tensor};

impl<B, const D: usize> Tensor<B, D, Bool>
where
    B: Backend,
{
    /// Create a boolean tensor from data.
    pub fn from_bool(data: Data<bool, D>) -> Self {
        Self::new(B::bool_from_data(data, &B::Device::default()))
    }

    /// Create a boolean tensor from data on the given device.
    pub fn from_bool_device(data: Data<bool, D>, device: &B::Device) -> Self {
        Self::new(B::bool_from_data(data, device))
    }

    /// Convert the bool tensor into an int tensor.
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::bool_into_int(self.primitive))
    }

    /// Convert the bool tensor into an float tensor.
    pub fn float(self) -> Tensor<B, D> {
        Tensor::new(B::bool_into_float(self.primitive))
    }
}
