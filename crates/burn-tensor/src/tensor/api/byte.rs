use crate::{backend::Backend, Float, Int, Tensor, TensorData, TensorPrimitive};

use super::Byte;

impl<const D: usize, B> Tensor<B, D, Byte>
where
    B: Backend,
{
    /// Create a tensor from bytes (u8), placing it on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int, Byte};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let _x: Tensor<B, 1, Byte> = Tensor::from_bytes([1, 2], &device);
    ///     let _y: Tensor<B, 2, Byte> = Tensor::from_bytes([[1, 2], [3, 4]], &device);
    /// }
    /// ```
    pub fn from_bytes<A: Into<TensorData>>(bytes: A, device: &B::Device) -> Self {
        Self::from_data(bytes.into().convert::<u8>(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Float.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let byte_tensor = Tensor::<B, 1, Int>::arange(0..5, &device).byte();
    ///     let float_tensor = byte_tensor.float();
    /// }
    /// ```
    pub fn float(self) -> Tensor<B, D, Float> {
        Tensor::new(TensorPrimitive::Float(B::byte_into_float(self.primitive)))
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Int.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let byte_tensor = Tensor::<B, 1, Int>::arange(0..5, &device).byte();
    ///     let int_tensor = byte_tensor.int();
    /// }
    /// ```
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::byte_into_int(self.primitive))
    }
}
