use crate::{backend::Backend, Data, Float, Int, Tensor};
use core::ops::Range;

impl<B> Tensor<B, 1, Int>
where
    B: Backend,
{
    /// Returns a new integer tensor on the default device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    pub fn arange(range: Range<usize>) -> Self {
        Tensor::new(B::arange(range, &B::Device::default()))
    }

    /// Returns a new integer tensor on the default device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `step` - The step between each value.
    pub fn arange_step(range: Range<usize>, step: usize) -> Self {
        Tensor::new(B::arange_step(range, step, &B::Device::default()))
    }

    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `device` - The device to create the tensor on.
    pub fn arange_device(range: Range<usize>, device: &B::Device) -> Self {
        Tensor::new(B::arange(range, device))
    }

    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `step` - The step between each value.
    pub fn arange_step_device(range: Range<usize>, step: usize, device: &B::Device) -> Self {
        Tensor::new(B::arange_step(range, step, device))
    }
}

impl<const D: usize, B> Tensor<B, D, Int>
where
    B: Backend,
{
    /// Create a tensor from integers (i32).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///     let _x: Tensor<B, 1, Int> = Tensor::from_ints([1, 2]);
    ///     let _y: Tensor<B, 2, Int> = Tensor::from_ints([[1, 2], [3, 4]]);
    /// }
    /// ```
    pub fn from_ints<A: Into<Data<i32, D>>>(ints: A) -> Self {
        Self::from_data(ints.into().convert())
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// casted to Float.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///     let int_tensor = Tensor::<B, 1, Int>::arange(0..5);
    ///     let float_tensor = int_tensor.float();
    /// }
    /// ```
    pub fn float(self) -> Tensor<B, D, Float> {
        Tensor::new(B::int_into_float(self.primitive))
    }
}
