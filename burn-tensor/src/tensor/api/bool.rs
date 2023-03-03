use crate::{backend::Backend, Bool, Data, Tensor};

impl<B, const D: usize> Tensor<B, D, Bool>
where
    B: Backend,
{
    /// Create a boolean tensor from data.
    pub fn from_bool(data: Data<bool, D>) -> Self {
        Self::new(B::from_data_bool(data, &B::Device::default()))
    }

    /// Create a boolean tensor from data on the given device.
    pub fn from_bool_device(data: Data<bool, D>, device: &B::Device) -> Self {
        Self::new(B::from_data_bool(data, device))
    }
}
