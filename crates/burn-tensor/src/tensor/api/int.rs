use crate::check;
use crate::check::TensorCheck;
use crate::{
    backend::Backend, cartesian_grid, Float, Int, Shape, Tensor, TensorData, TensorPrimitive,
};

use core::ops::Range;

impl<B> Tensor<B, 1, Int>
where
    B: Backend,
{
    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `device` - The device to create the tensor on.
    pub fn arange(range: Range<i64>, device: &B::Device) -> Self {
        Tensor::new(B::int_arange(range, device))
    }

    /// Returns a new integer tensor on the specified device.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to generate.
    /// * `step` - The step between each value.
    pub fn arange_step(range: Range<i64>, step: usize, device: &B::Device) -> Self {
        Tensor::new(B::int_arange_step(range, step, device))
    }

    /// Create a one hot tensor from an index tensor.
    ///
    /// # Arguments
    ///
    /// * `num_classes` - The number of classes to use in encoding.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let indices: Tensor<B, 1, Int> = Tensor::from_ints([0, 1, 2, 3], &device);
    ///     let one_hot = indices.one_hot(4);
    ///     println!("{}", one_hot.to_data());
    ///     // [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    /// }
    /// ```
    pub fn one_hot(self, num_classes: usize) -> Tensor<B, 2, Int> {
        check!(TensorCheck::one_hot_tensor(self.clone(), num_classes));
        let [num_samples] = self.dims();
        let indices = self.unsqueeze();
        let values = indices.ones_like();
        Tensor::zeros([num_samples, num_samples], &indices.device()).scatter(1, indices, values)
    }
}

impl<const D: usize, B> Tensor<B, D, Int>
where
    B: Backend,
{
    /// Create a tensor from integers (i32), placing it on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let _x: Tensor<B, 1, Int> = Tensor::from_ints([1, 2], &device);
    ///     let _y: Tensor<B, 2, Int> = Tensor::from_ints([[1, 2], [3, 4]], &device);
    /// }
    /// ```
    pub fn from_ints<A: Into<TensorData>>(ints: A, device: &B::Device) -> Self {
        Self::from_data(ints.into().convert::<i32>(), device)
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
    ///     let int_tensor = Tensor::<B, 1, Int>::arange(0..5, &device);
    ///     let float_tensor = int_tensor.float();
    /// }
    /// ```
    pub fn float(self) -> Tensor<B, D, Float> {
        Tensor::new(TensorPrimitive::Float(B::int_into_float(self.primitive)))
    }

    /// Generates a cartesian grid for the given tensor shape on the specified device.
    /// The generated tensor is of dimension `D2 = D + 1`, where each element at dimension D contains the cartesian grid coordinates for that element.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape specifying the dimensions of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Panics
    ///
    /// Panics if `D2` is not equal to `D+1`.
    ///
    /// # Examples
    ///
    /// ```rust
    ///    use burn_tensor::Int;
    ///    use burn_tensor::{backend::Backend, Shape, Tensor};
    ///    fn example<B: Backend>() {
    ///        let device = Default::default();
    ///        let result: Tensor<B, 3, _> = Tensor::<B, 2, Int>::cartesian_grid([2, 3], &device);
    ///        println!("{}", result);
    ///    }
    /// ```
    pub fn cartesian_grid<S: Into<Shape>, const D2: usize>(
        shape: S,
        device: &B::Device,
    ) -> Tensor<B, D2, Int> {
        cartesian_grid::<B, S, D, D2>(shape, device)
    }
}
