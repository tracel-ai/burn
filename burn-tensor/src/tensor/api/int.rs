use crate::{backend::Backend, Data, Int, Tensor};
use core::ops::Range;

impl<B> Tensor<B, 1, Int>
where
    B: Backend,
{
    /// Returns a new integer tensor on the default device which values are generated from the given range.
    pub fn arange(range: Range<usize>) -> Self {
        Tensor::new(B::arange(range, &B::Device::default()))
    }
    /// Returns a new integer tensor on the specified device which values are generated from the given range.
    pub fn arange_device(range: Range<usize>, device: &B::Device) -> Self {
        Tensor::new(B::arange(range, device))
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
}
