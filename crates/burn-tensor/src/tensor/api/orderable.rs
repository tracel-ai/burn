use burn_backend::{
    Backend, ElementComparison, ElementConversion, Scalar,
    tensor::{Bool, IndexingUpdateOp, Int, Ordered},
};
use burn_std::AsIndex;

use crate::check;
use crate::{Tensor, check::TensorCheck};

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Ordered<B>,
    K::Elem: ElementComparison,
{
    /// Create a one hot tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>(){
    ///     let device = Default::default();
    ///     let indices: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0, 2.0, 3.0], &device);
    ///     let one_hot: Tensor<B, 2> = indices.one_hot(4);
    ///     println!("{}", one_hot.to_data());
    ///     // [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    /// }
    /// ```
    pub fn one_hot<const D2: usize>(self, num_classes: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::one_hot_tensor(self.clone(), num_classes));
        self.one_hot_fill(num_classes, 1.0, 0.0, -1)
    }

    /// Create a one-hot encoded tensor with configurable `num_classes`, `on_value`, `off_value`, and `axis` including high-ranked tensors.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: The number of classes for the one-hot encoding, which defines the size of the one-hot dimension.
    /// * `on_value`: The value to assign for active positions (corresponding to indices).
    /// * `off_value`: The value to assign for inactive positions.
    /// * `axis`: The axis along which the one-hot dimension is added. Supports negative indexing.
    ///
    /// # Returns
    ///
    /// A tensor with one additional dimension for the one-hot encoding, where active positions are filled with `on_value` and others with `off_value`.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Float};
    /// fn example<B: Backend<FloatElem: From<f32>>>() {
    ///     let device = B::Device::default();
    ///     let indices: Tensor<B, 2, Float> = Tensor::from_floats([[0., 2.], [1., -1.]], &device);
    ///     // One-hot encoding
    ///     let tensor:Tensor<B, 3, Float> = indices.one_hot_fill(3, 5.0.into(), 0.0.into(), -1);
    ///     println!("{tensor}");
    ///     // [[[5.0, 0.0, 0.0],
    ///     // [0.0, 0.0, 5.0]],
    ///     // [[0.0, 5.0, 0.0],
    ///     // [0.0, 0.0, 5.0]]]
    /// }
    /// ```
    pub fn one_hot_fill<const D2: usize>(
        self,
        num_classes: usize,
        on_value: f32,
        off_value: f32,
        axis: i64,
    ) -> Tensor<B, D2, K> {
        check!(TensorCheck::one_hot_tensor_rank::<D, D2>());
        // Initialize shape from the current tensor dimensions and prepare for modification
        let mut shape = self.shape();
        let device = self.device();
        let rank = self.dims().len();

        // Adjust negative axis to a positive index
        let axis = if axis < 0 {
            axis + rank as i64 + 1
        } else {
            axis
        };

        // Ensure axis is within valid range
        if axis < 0 || axis > rank as i64 {
            panic!("Axis out of range. Accepted range is [-r-1, r] where r = rank(indices).");
        }
        // Convert the input tensor to integer indices
        let indices: Tensor<B, D, Int> =
            Tensor::from_data(self.to_data().convert::<i64>(), &device);
        // Insert the new dimension for the one-hot representation
        shape.insert(axis as usize, num_classes);
        // Adjust indices to valid range and handle invalid indices
        let adjusted_indices = indices
            .clone()
            .mask_fill(self.clone().lower_elem(0), num_classes as i64) // Handle negative indices
            .add(indices.clone().mask_fill(self.clone().greater_elem(0), 0)); // Handle positive indices
        // Unsqueeze the indices tensor along the specified axis
        let indices_unsqueezed: Tensor<B, D2, Int> = adjusted_indices.unsqueeze_dim(axis as usize);

        // Initialize the output tensor with the off_value
        let output = Tensor::full(shape.clone(), off_value, &device);

        // Prepare scatter tensor for on_value and off_value adjustments
        let scatter_on_values = Tensor::full(indices_unsqueezed.shape(), on_value, &device)
            - Tensor::full(indices_unsqueezed.shape(), off_value, &self.device());

        // Scatter on_value at the appropriate indices to create the one-hot representation
        output.scatter(
            axis as usize,
            indices_unsqueezed,
            scatter_on_values,
            IndexingUpdateOp::Add,
        )
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor2 = Tensor::<B, 2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///   let tensor = tensor1.greater(tensor2);
    ///   println!("{tensor}");
    ///   // [[false, false, false], [true, true, true]]
    /// }
    /// ```
    pub fn greater(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater", &self, &other));
        Tensor::new(K::greater(self.primitive, other.primitive))
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.greater_equal(tensor2);
    ///    println!("{tensor}");
    ///    // [[true, false, false], [true, true, true]]
    /// }
    /// ```
    pub fn greater_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater_equal", &self, &other));
        Tensor::new(K::greater_equal(self.primitive, other.primitive))
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.lower(tensor2);
    ///    println!("{tensor}");
    ///    // [[false, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower", &self, &other));
        Tensor::new(K::lower(self.primitive, other.primitive))
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.lower_equal(tensor2);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower_equal", &self, &other));
        Tensor::new(K::lower_equal(self.primitive, other.primitive))
    }

    /// Applies greater than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.greater_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[false, false, true], [true, true, true]]
    /// }
    /// ```
    pub fn greater_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::greater_elem(self.primitive, other))
    }

    /// Applies greater-equal than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.greater_equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[false, false, true], [true, true, true]]
    /// }
    /// ```
    pub fn greater_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::greater_equal_elem(self.primitive, other))
    }

    /// Applies lower than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///     let tensor = tensor.lower_elem(3.0);
    ///     println!("{tensor}");
    ///     // [[true, true, false], [false, false, false]]
    /// }
    /// ```
    pub fn lower_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::lower_elem(self.primitive, other))
    }

    /// Applies lower-equal than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.lower_equal_elem(3.0);
    ///    println!("{tensor}");
    ///    // [[true, true, true], [false, false, false]]
    /// }
    /// ```
    pub fn lower_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::lower_equal_elem(self.primitive, other))
    }

    /// Applies the argmax function along the given dimension and returns an integer tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]), &device);
    ///     let tensor = tensor.argmax(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmax(self, dim: usize) -> Tensor<B, D, Int> {
        Tensor::new(K::argmax(self.primitive, dim))
    }

    /// Find the maximum value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max();
    ///   println!("{tensor}");
    ///   // [9.0]
    /// }
    /// ```
    pub fn max(self) -> Tensor<B, 1, K> {
        Tensor::new(K::max(self.primitive))
    }

    /// Find the maximum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let (tensor, index) = tensor.max_dim_with_indices(0);
    ///    // [[5.0, 9.0, 6.0]]
    ///    println!("{tensor}");
    ///    // [[1, 1, 1]]
    ///    println!("{index}");
    /// }
    /// ```
    pub fn max_dim_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<B, D, Int>) {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));

        let (tensor, index) = K::max_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Find the maximum absolute value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -7.0, 3.0], [5.0, -1.0, 6.0]], &device);
    ///   let tensor = tensor.max_abs();
    ///   println!("{tensor}");
    ///   // [7.0]
    /// }
    /// ```
    pub fn max_abs(self) -> Tensor<B, 1, K> {
        Tensor::new(K::max_abs(self.primitive))
    }

    /// Finds the maximum pair wise values with another tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find maximum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the maximum value found
    /// in the input tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.max_pair(tensor2);
    ///    println!("{tensor}");
    ///    // [[2.0, 3.0, 4.0], [5.0, 9.0, 6.0]]
    /// }
    /// ```
    pub fn max_pair(self, other: Self) -> Self {
        let mask = self.clone().lower(other.clone());
        self.mask_where(mask, other)
    }

    /// Find the maximum absolute value along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements,
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimension will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max_dim(0);
    ///   println!("{tensor}");
    ///   // [[5.0, 9.0, 6.0]]
    /// }
    /// ```
    pub fn max_abs_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("MaxAbs", dim));

        Tensor::new(K::max_abs_dim(self.primitive, dim))
    }

    /// Find the maximum absolute value along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions or axes along which to aggregate the elements,
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max_abs_dims(&[0, 1]);
    ///   println!("{tensor}");
    ///   // [[9.0]]
    /// }
    /// ```
    pub fn max_abs_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter()
            .fold(self, |tensor, &dim| tensor.max_abs_dim(dim))
    }

    /// Applies the argmin function along the given dimension and returns an integer tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]), &device);
    ///     let tensor = tensor.argmin(1);
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [2, 1, 3] }
    /// }
    /// ```
    pub fn argmin(self, dim: usize) -> Tensor<B, D, Int> {
        Tensor::new(K::argmin(self.primitive, dim))
    }

    /// Find the minimum value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.min();
    ///    println!("{tensor}");
    ///    // [-2.0]
    /// }
    /// ```
    pub fn min(self) -> Tensor<B, 1, K> {
        Tensor::new(K::min(self.primitive))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements;
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimension will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.min_dim(0);
    ///    println!("{tensor}");
    ///    // [[1.0, -2.0, 3.0]]
    /// }
    /// ```
    pub fn min_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));
        Tensor::new(K::min_dim(self.primitive, dim))
    }

    /// Find the minimum value along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions or axes along which to aggregate the elements;
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.min_dims(&[0, 1]);
    ///   println!("{tensor}");
    ///   // [[-2.0]]
    /// }
    /// ```
    pub fn min_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.min_dim(dim))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[7.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let (tensor, index) = tensor.min_dim_with_indices(0);
    ///    println!("{tensor}");
    ///    // [[5.0, -2.0, 3.0]]
    ///    println!("{}", index);
    ///    // [[1, 0, 0]]
    /// }
    /// ```
    pub fn min_dim_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<B, D, Int>) {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));

        let (tensor, index) = K::min_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Finds the minimum pair wise values with another tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find minimum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the minimum value found
    /// between each element of the two source tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor1 = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor2 = Tensor::<B, 2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    ///    let tensor = tensor1.min_pair(tensor2);
    ///    println!("{tensor}");
    ///    // [[1.0, -2.0, 3.0], [1.0, 2.0, 3.0]]
    /// }
    pub fn min_pair(self, other: Self) -> Self {
        let mask = other.clone().lower(self.clone());
        self.mask_where(mask, other)
    }

    /// Clamp element wise between the given min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped between the given min and max values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9]
    ///    ],
    ///    &device);
    ///    let tensor = tensor.clamp(2, 6);
    ///    println!("{tensor}");
    ///    // [[2, 2, 3], [4, 5, 6], [6, 6, 6]]
    /// }
    /// ```
    pub fn clamp<E: ElementConversion>(self, min: E, max: E) -> Self {
        let dtype = self.dtype();
        Self::new(K::clamp(
            self.primitive,
            Scalar::new(min, &dtype),
            Scalar::new(max, &dtype),
        ))
    }

    /// Clamp element wise under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped under the given min value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ///    &device);
    ///    let tensor = tensor.clamp_min(4);
    ///    println!("{tensor}");
    ///    // [[4, 4, 4], [4, 5, 6], [7, 8, 9]]
    /// }
    /// ```
    pub fn clamp_min<E: ElementConversion>(self, min: E) -> Self {
        let min = Scalar::new(min, &self.dtype());
        Self::new(K::clamp_min(self.primitive, min))
    }

    /// Clamp element wise over a maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped over the given max value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ///    &device);
    ///    let tensor = tensor.clamp_max(5);
    ///    println!("{tensor}");
    ///    // [[1, 2, 3], [4, 5, 5], [5, 5, 5]]
    /// }
    /// ```
    pub fn clamp_max<E: ElementConversion>(self, max: E) -> Self {
        let max = Scalar::new(max, &self.dtype());
        Self::new(K::clamp_max(self.primitive, max))
    }

    /// Computes the cumulative minimum of elements along the given *dimension* or *axis*.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative minimum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[3.0, 5.0, 2.0], [4.0, 1.0, 6.0]], &device);
    ///    let result = tensor.clone().cummin(0);
    ///    println!("{result}");
    ///    // [[3.0, 5.0, 2.0], [3.0, 1.0, 2.0]]
    ///    let result = tensor.cummin(1);
    ///    println!("{result}");
    ///    // [[3.0, 3.0, 2.0], [4.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn cummin(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("CumMin", dim));
        Self::new(K::cummin(self.primitive, dim))
    }

    /// Computes the cumulative maximum of elements along the given *dimension* or *axis*.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative maximum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[3.0, 1.0, 2.0], [4.0, 5.0, 2.0]], &device);
    ///    let result = tensor.clone().cummax(0);
    ///    println!("{result}");
    ///    // [[3.0, 1.0, 2.0], [4.0, 5.0, 2.0]]
    ///    let result = tensor.cummax(1);
    ///    println!("{result}");
    ///    // [[3.0, 3.0, 3.0], [4.0, 5.0, 5.0]]
    /// }
    /// ```
    pub fn cummax(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("CumMax", dim));
        Self::new(K::cummax(self.primitive, dim))
    }
    /// Find the maximum value along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements;
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimension will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max_dim(0);
    ///   println!("{tensor}");
    ///   // [[5.0, 9.0, 6.0]]
    /// }
    /// ```
    pub fn max_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = dim.expect_dim_index(D);
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));
        Tensor::new(K::max_dim(self.primitive, dim))
    }

    /// Find the maximum value along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions or axis along which to aggregate the elements;
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.max_dims(&[0, 1]);
    ///   println!("{tensor}");
    ///   // [[9.0]]
    /// }
    /// ```
    pub fn max_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.max_dim(dim))
    }
}
