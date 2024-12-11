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
        let indices = self.unsqueeze_dim(1);
        let values = indices.ones_like();
        Tensor::zeros([num_samples, num_classes], &indices.device()).scatter(1, indices, values)
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

    /// Create a one-hot encoded tensor with configurable `on_value`, `off_value`, and `axis`.
    ///
    /// # Arguments
    ///
    /// * `depth` - The number of classes for one-hot encoding.
    /// * `on_value` - The value to use for the "on" positions.
    /// * `off_value` - The value to use for the "off" positions.
    /// * `axis` - The axis along which to perform one-hot encoding.
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let expected: Tensor<B, 2, Int> = Tensor::from_ints([[5, 0, 0], [0, 0, 5], [0, 5, 0], [0, 0, 5]], &device);
    ///     let indices: Tensor<B, 2, Int> = Tensor::from_ints([[0, 2], [1, -1]], &device);
    ///     // One-hot encoding
    ///     let result = indices.one_hot_with_axis_and_values(3, 5, 0, -1);
    ///     assert_eq!(expected.to_data(), result.to_data());
    /// }
    /// ```
    pub fn one_hot_with_axis_and_values(
        self,
        depth: usize,
        on_value: i64,
        off_value: i64,
        axis: i64,
    ) -> Tensor<B, D, Int> {

    let mut shape = self.shape().dims::<D>().to_vec();
    let rank = self.dims().len();
    let axis = if axis < 0 {
        axis + rank as i64 + 1 // Convert negative axis to positive index
    } else {
        axis
    };
    if axis < 0 || axis > rank as i64 {
        panic!("Axis out of range. Accepted range is [-r-1, r] where r = rank(indices).");
    }
    shape.insert(axis as usize, depth);
    let condition1 = self.clone().greater_elem(-1 * depth as i64).int();
    let condition2 = self.clone().lower_elem(depth as i64).int();
    let valid_mask = condition1.mul(condition2).bool().bool_not();
    let adjusted_indices = self
        .clone()
        .mask_fill(self.clone().lower_elem(0), depth as i64)
        .add(
            self
                .clone()
                .mask_fill(self.clone().greater_elem(0), 0),
        );

    let valid_indices = adjusted_indices.mask_fill(valid_mask, off_value);
    let indices_unsqueezed = valid_indices.unsqueeze_dim(axis as usize);

    let output= Tensor::full(shape.clone(), off_value, &self.device());
    let scatter_on_values = Tensor::full(indices_unsqueezed.shape(), on_value, &self.device());
    let scatter_off_values = Tensor::full(indices_unsqueezed.shape(), -off_value, &self.device());

    output
        .scatter(axis as usize, indices_unsqueezed.clone(), scatter_on_values)
        .scatter(axis as usize, indices_unsqueezed, scatter_off_values)
    }
}
