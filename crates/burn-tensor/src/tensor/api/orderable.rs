use alloc::{boxed::Box, vec::Vec};
use burn_backend::{ElementConversion, Scalar};
use burn_std::{AsIndex, IndexingUpdateOp};

use crate::check::unwrap_dim_index;
use crate::kind::Ordered;
use crate::{Bool, Int, TensorData, check};
use crate::{Tensor, check::TensorCheck};

impl<const D: usize, K> Tensor<D, K>
where
    K: Ordered,
{
    /// Sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Returns
    ///
    /// A new tensor with the elements sorted in ascending order along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let sorted = tensor.clone().sort(0);
    /// println!("{sorted}");
    /// // [[5.0, -2.0, 3.0], [12.0, 3.0, 6.0]]
    /// let sorted = tensor.sort(1);
    /// println!("{sorted}");
    /// // [[-2.0, 3.0, 12.0], [3.0, 5.0, 6.0]]
    /// ```
    pub fn sort<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Sort");
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ false))
    }

    /// Sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Returns
    ///
    /// A new tensor with the elements sorted in descending order along the given dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let sorted = tensor.clone().sort_descending(0);
    /// println!("{sorted}");
    /// // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    /// let sorted = tensor.sort_descending(1);
    /// println!("{sorted}");
    /// // [[12.0, 3.0, -2.0], [6.0, 5.0, 3.0]]
    /// ```
    pub fn sort_descending<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Sort Descending");
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ true))
    }

    /// Sort the elements by value in ascending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Returns
    ///
    /// A tuple containing the sorted tensor and the indices tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let (tensor, indices) = tensor.sort_with_indices(0);
    /// println!("{tensor}");
    /// // [[5.0, -2.0, 3.0], [12.0, 3.0, 6.0]]
    /// println!("{}", indices);
    /// // [[1, 0, 0], [0, 1, 1]]
    /// ```
    pub fn sort_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<D, Int>) {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Sort With Indices");
        let (values, indices) =
            K::sort_with_indices(self.primitive, dim, /*descending*/ false);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Sort the elements by value in descending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let (tensor, indices) = tensor.sort_descending_with_indices(0);
    /// println!("{tensor}");
    /// // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    /// println!("{}", indices);
    /// // [[0, 1, 1], [1, 0, 0]]
    /// ```
    pub fn sort_descending_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<D, Int>) {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Sort Descending With Indices");
        let (values, indices) = K::sort_with_indices(self.primitive, dim, /*descending*/ true);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Returns the indices that sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let tensor = tensor.argsort(0);
    /// println!("{tensor}");
    /// // [[1, 0, 0], [0, 1, 1]]
    /// ```
    pub fn argsort<I: AsIndex>(self, dim: I) -> Tensor<D, Int> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Argsort");
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ false))
    }

    /// Returns the indices that sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let indices = tensor.clone().argsort_descending(0);
    /// println!("{indices}");
    /// // [[0, 1, 1], [1, 0, 0]]
    /// let indices = tensor.argsort_descending(1);
    /// println!("{indices}");
    /// // [[0, 2, 1], [2, 0, 1]]
    /// ```
    pub fn argsort_descending<I: AsIndex>(self, dim: I) -> Tensor<D, Int> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Argsort Descending");
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ true))
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of elements to return.
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Returns
    ///
    /// A new tensor with the `k` largest elements along the given dimension.
    ///
    /// With autodiff, gradients are propagated to the input positions selected during the forward
    /// pass. All other input positions receive zero gradient. When values are tied, the gradient
    /// follows the indices selected by the backend.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let topk = tensor.clone().topk(2, 0);
    /// println!("{topk}");
    /// // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    /// let topk = tensor.topk(1, 1);
    /// println!("{topk}");
    /// // [[12.0], [6.0]]
    /// ```
    pub fn topk<I: AsIndex>(self, k: usize, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Top K");
        check!(TensorCheck::topk("Top K", k, dim, &self.shape()));
        Tensor::new(K::topk(self.primitive, dim, k))
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    /// Also returns the indices.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of elements to return.
    /// * `dim` - The dimension to sort along.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// With autodiff, gradients from the returned values are propagated to the input positions
    /// identified by the returned indices. All other input positions receive zero gradient.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[12.0, -2.0, 3.0], [5.0, 3.0, 6.0]], &device);
    /// let (tensor, indices) = tensor.topk_with_indices(2, 0);
    /// println!("{tensor}");
    /// // [[12.0, 3.0, 6.0], [5.0, -2.0, 3.0]]
    /// println!("{}", indices);
    /// // [[0, 1, 1], [1, 0, 0]]
    /// let (tensor, indices) = tensor.topk_with_indices(1, 1);
    /// println!("{tensor}");
    /// // [[12.0], [6.0]]
    /// println!("{indices}");
    /// // [[0], [2]]
    /// ```
    pub fn topk_with_indices<I: AsIndex>(self, k: usize, dim: I) -> (Self, Tensor<D, Int>) {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Top K With Indices");
        check!(TensorCheck::topk(
            "Top K With Indices",
            k,
            dim,
            &self.shape()
        ));
        let (values, indices) = K::topk_with_indices(self.primitive, dim, k);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Create a one hot tensor.
    ///
    /// # Panics
    ///
    /// If an index is outside `[-num_classes, num_classes)`, if `num_classes` is less than 2, or if
    /// the tensor is quantized. Use [`one_hot_fill`](Tensor::one_hot_fill) to get an `off_value` row
    /// for out-of-range indices instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::Tensor;
    ///
    /// fn example(){
    ///     let device = Default::default();
    ///     let indices: Tensor<1> = Tensor::from_floats([0.0, 1.0, 2.0, 3.0], &device);
    ///     let one_hot: Tensor<2> = indices.one_hot(4);
    ///     println!("{}", one_hot.to_data());
    ///     // [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    /// }
    /// ```
    pub fn one_hot<const D2: usize>(self, num_classes: usize) -> Tensor<D2, K> {
        check!(TensorCheck::quantized_unsupported("One Hot", self.dtype()));
        let classes = one_hot_classes(&self.to_data(), num_classes);
        check!(TensorCheck::one_hot_tensor(&self, num_classes, &classes));
        check!(TensorCheck::one_hot_tensor_rank::<D, D2>());
        self.one_hot_from_classes(num_classes, 1.0, 0.0, D, &classes)
    }

    /// Create a one-hot encoded tensor with configurable `num_classes`, `on_value`, `off_value`, and `axis` including high-ranked tensors.
    ///
    /// # Arguments
    ///
    /// * `num_classes`: The number of classes for the one-hot encoding, which defines the size of the one-hot dimension.
    /// * `on_value`: The value to assign for active positions (corresponding to indices).
    /// * `off_value`: The value to assign for inactive positions.
    /// * `axis`: The axis along which the one-hot dimension is added.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Returns
    ///
    /// A tensor with one additional dimension for the one-hot encoding, where active positions are filled with `on_value` and others with `off_value`.
    ///
    /// Indices outside `[-num_classes, num_classes)` produce a row filled with `off_value`, as in
    /// ONNX `OneHot`. Float indices are truncated toward zero first; NaN and infinities are out of
    /// range. With `num_classes == 0`, every index is out of range and the one-hot
    /// dimension is empty. Use [`one_hot`](Tensor::one_hot) to reject out-of-range indices instead.
    ///
    /// # Panics
    ///
    /// If the tensor is quantized, since lossy quantized values can't be trusted as class indices.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::{Tensor, Float};
    /// let device = Default::default();
    /// let indices: Tensor<2, Float> = Tensor::from_floats([[0., 2.], [1., -1.]], &device);
    /// // One-hot encoding
    /// let tensor: Tensor<3, Float> = indices.one_hot_fill(3, 5.0.into(), 0.0.into(), -1);
    /// println!("{tensor}");
    /// // [[[5.0, 0.0, 0.0],
    /// // [0.0, 0.0, 5.0]],
    /// // [[0.0, 5.0, 0.0],
    /// // [0.0, 0.0, 5.0]]]
    /// ```
    pub fn one_hot_fill<const D2: usize>(
        self,
        num_classes: usize,
        on_value: f32,
        off_value: f32,
        axis: impl AsIndex,
    ) -> Tensor<D2, K> {
        check!(TensorCheck::one_hot_tensor_rank::<D, D2>());
        check!(TensorCheck::quantized_unsupported("One Hot", self.dtype()));
        let axis = unwrap_dim_index(axis.try_dim_index(D + 1), "One Hot");

        let classes = if num_classes == 0 {
            Vec::new()
        } else {
            one_hot_classes(&self.to_data(), num_classes)
        };
        self.one_hot_from_classes(num_classes, on_value, off_value, axis, &classes)
    }

    fn one_hot_from_classes<const D2: usize>(
        self,
        num_classes: usize,
        on_value: f32,
        off_value: f32,
        axis: usize,
        classes: &[Option<usize>],
    ) -> Tensor<D2, K> {
        // Initialize shape from the current tensor dimensions and prepare for modification
        let index_shape = self.shape();
        let mut shape = index_shape.clone();
        let device = self.device();
        let dtype = self.dtype();

        // Insert the new dimension for the one-hot representation
        shape.insert(axis, num_classes);

        // Initialize the output tensor with the off_value
        let output = Tensor::full(shape.clone(), off_value, (&device, dtype));

        // Every index is out of range: the one-hot dimension is empty and there is nothing to scatter
        if num_classes == 0 {
            return output;
        }

        // Out-of-range positions get class 0 and scatter off_value there, so their row is unchanged
        let out_of_range: Vec<bool> = classes.iter().map(Option::is_none).collect();
        let class_indices: Vec<i64> = classes.iter().map(|c| c.unwrap_or(0) as i64).collect();
        let adjusted_indices: Tensor<D, Int> =
            Tensor::from_data(TensorData::new(class_indices, index_shape.clone()), &device);
        let out_of_range: Tensor<D, Bool> =
            Tensor::from_data(TensorData::new(out_of_range, index_shape), &device);

        // Unsqueeze the indices tensor along the specified axis
        let indices_unsqueezed: Tensor<D2, Int> = adjusted_indices.unsqueeze_dim(axis);

        // Out-of-range positions scatter off_value, so their row stays unchanged
        let on_values = Tensor::full(indices_unsqueezed.shape(), on_value, (&device, dtype))
            .mask_fill(out_of_range.unsqueeze_dim(axis), off_value);

        // Scatter on_value at the appropriate indices to create the one-hot representation
        output.scatter(
            axis,
            indices_unsqueezed,
            on_values,
            IndexingUpdateOp::Assign,
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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.greater(tensor2);
    /// println!("{tensor}");
    /// // [[false, false, false], [true, true, true]]
    /// ```
    pub fn greater(self, other: Self) -> Tensor<D, Bool> {
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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.greater_equal(tensor2);
    /// println!("{tensor}");
    /// // [[true, false, false], [true, true, true]]
    /// ```
    pub fn greater_equal(self, other: Self) -> Tensor<D, Bool> {
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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.lower(tensor2);
    /// println!("{tensor}");
    /// // [[false, true, true], [false, false, false]]
    /// ```
    pub fn lower(self, other: Self) -> Tensor<D, Bool> {
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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[1.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.lower_equal(tensor2);
    /// println!("{tensor}");
    /// // [[true, true, true], [false, false, false]]
    /// ```
    pub fn lower_equal(self, other: Self) -> Tensor<D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower_equal", &self, &other));
        Tensor::new(K::lower_equal(self.primitive, other.primitive))
    }

    /// Applies greater than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.greater_scalar(3.0);
    /// println!("{tensor}");
    /// // [[false, false, true], [true, true, true]]
    /// ```
    pub fn greater_scalar<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::greater_scalar(self.primitive, other))
    }

    /// Applies greater-equal than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.greater_equal_scalar(3.0);
    /// println!("{tensor}");
    /// // [[false, false, true], [true, true, true]]
    /// ```
    pub fn greater_equal_scalar<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::greater_equal_scalar(self.primitive, other))
    }

    /// Applies lower than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.lower_scalar(3.0);
    /// println!("{tensor}");
    /// // [[true, true, false], [false, false, false]]
    /// ```
    pub fn lower_scalar<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::lower_scalar(self.primitive, other))
    }

    /// Applies lower-equal than `other` comparison and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to compare.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.lower_equal_scalar(3.0);
    /// println!("{tensor}");
    /// // [[true, true, true], [false, false, false]]
    /// ```
    pub fn lower_equal_scalar<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        let other = Scalar::new(other, &self.dtype());
        Tensor::new(K::lower_equal_scalar(self.primitive, other))
    }

    /// Alias for [greater_scalar](Self::greater_scalar).
    pub fn greater_elem<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        self.greater_scalar(other)
    }

    /// Alias for [greater_equal_scalar](Self::greater_equal_scalar).
    pub fn greater_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        self.greater_equal_scalar(other)
    }

    /// Alias for [lower_scalar](Self::lower_scalar).
    pub fn lower_elem<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        self.lower_scalar(other)
    }

    /// Alias for [lower_equal_scalar](Self::lower_equal_scalar).
    pub fn lower_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<D, Bool> {
        self.lower_equal_scalar(other)
    }

    /// Applies the argmax function along the given dimension and returns an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to find the maximum value.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # NaN behavior
    ///
    /// For floating-point tensors, NaNs take precedence over non-NaN values. If a reduced slice
    /// contains multiple NaNs, the lowest coordinate along `dim` is returned. Non-NaN ties also
    /// return the lowest coordinate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<3>::ones(Shape::new([2, 3, 3]), &device);
    /// let tensor = tensor.argmax(1);
    /// println!("{:?}", tensor.shape());
    /// // Shape { dims: [2, 1, 3] }
    ///
    /// let tensor = Tensor::<1>::from_data([3.0, f32::NAN, f32::NAN], &device);
    /// let index: i32 = tensor.argmax(0).into_scalar();
    /// assert_eq!(index, 1);
    /// ```
    pub fn argmax(self, dim: impl AsIndex) -> Tensor<D, Int> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Argmax");
        Tensor::new(K::argmax(self.primitive, dim))
    }

    /// Applies the argtopk function along the given dimension and returns an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of indices to return.
    /// * `dim` - The dimension along which to find the largest values.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<3>::ones(Shape::new([2, 3, 3]), &device);
    /// let tensor = tensor.argtopk(1, 2);
    /// println!("{:?}", tensor.shape());
    /// ```
    pub fn argtopk(self, k: usize, dim: impl AsIndex) -> Tensor<D, Int> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Argtopk");
        assert!(self.shape()[dim] > k);
        Tensor::new(K::argtopk(self.primitive, dim, k))
    }

    /// Find the maximum value.
    ///
    /// For floating-point tensors, the result is NaN if any element is NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.max();
    /// println!("{tensor}");
    /// // [9.0]
    ///
    /// let tensor = Tensor::<1>::from_data([1.0, f32::NAN, 3.0], &device);
    /// let value: f32 = tensor.max().into_scalar();
    /// assert!(value.is_nan());
    /// ```
    pub fn max(self) -> Tensor<1, K> {
        Tensor::new(K::max(self.primitive))
    }

    /// Find the maximum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// For floating-point tensors, a NaN in a reduced slice produces a NaN value. The returned
    /// index is the lowest coordinate containing NaN. Non-NaN ties also return the lowest
    /// coordinate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let (tensor, index) = tensor.max_dim_with_indices(0);
    /// // [[5.0, 9.0, 6.0]]
    /// println!("{tensor}");
    /// // [[1, 1, 1]]
    /// println!("{index}");
    /// ```
    pub fn max_dim_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<D, Int>) {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Max Dim With Indices");

        let (tensor, index) = K::max_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Find the maximum absolute value.
    ///
    /// For floating-point tensors, the result is NaN if any element is NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -7.0, 3.0], [5.0, -1.0, 6.0]], &device);
    /// let tensor = tensor.max_abs();
    /// println!("{tensor}");
    /// // [7.0]
    /// ```
    pub fn max_abs(self) -> Tensor<1, K> {
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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.max_pair(tensor2);
    /// println!("{tensor}");
    /// // [[2.0, 3.0, 4.0], [5.0, 9.0, 6.0]]
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
    /// For floating-point tensors, a reduced slice produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.max_dim(0);
    /// println!("{tensor}");
    /// // [[5.0, 9.0, 6.0]]
    /// ```
    pub fn max_abs_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Max Abs Dim");

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
    /// For floating-point tensors, a reduced region produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.max_abs_dims(&[0, 1]);
    /// println!("{tensor}");
    /// // [[9.0]]
    /// ```
    ///
    /// # Empty dimensions
    ///
    /// Unlike ordinary reductions such as [`Tensor::max_dims`], this is a composite
    /// operation equivalent to `self.abs().max_dims(dims)`. If `dims` is empty, no
    /// reduction is performed, but the elementwise absolute-value operation is
    /// still applied.
    pub fn max_abs_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        if dims.is_empty() {
            return self.abs();
        }
        dims.iter()
            .fold(self, |tensor, &dim| tensor.max_abs_dim(dim))
    }

    /// Applies the argmin function along the given dimension and returns an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to find the minimum value.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// # NaN behavior
    ///
    /// For floating-point tensors, NaNs take precedence over non-NaN values. If a reduced slice
    /// contains multiple NaNs, the lowest coordinate along `dim` is returned. Non-NaN ties also
    /// return the lowest coordinate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<3>::ones(Shape::new([2, 3, 3]), &device);
    /// let tensor = tensor.argmin(1);
    /// println!("{:?}", tensor.shape());
    /// // Shape { dims: [2, 1, 3] }
    /// ```
    pub fn argmin(self, dim: impl AsIndex) -> Tensor<D, Int> {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Argmin");
        Tensor::new(K::argmin(self.primitive, dim))
    }

    /// Find the minimum value.
    ///
    /// For floating-point tensors, the result is NaN if any element is NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.min();
    /// println!("{tensor}");
    /// // [-2.0]
    /// ```
    pub fn min(self) -> Tensor<1, K> {
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
    /// For floating-point tensors, a reduced slice produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.min_dim(0);
    /// println!("{tensor}");
    /// // [[1.0, -2.0, 3.0]]
    /// ```
    pub fn min_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Min Dim");
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
    /// For floating-point tensors, a reduced region produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.min_dims(&[0, 1]);
    /// println!("{tensor}");
    /// // [[-2.0]]
    /// ```
    pub fn min_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.min_dim(dim))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// Also returns the indices.
    ///
    /// For floating-point tensors, a NaN in a reduced slice produces a NaN value. The returned
    /// index is the lowest coordinate containing NaN. Non-NaN ties also return the lowest
    /// coordinate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[7.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let (tensor, index) = tensor.min_dim_with_indices(0);
    /// println!("{tensor}");
    /// // [[5.0, -2.0, 3.0]]
    /// println!("{}", index);
    /// // [[1, 0, 0]]
    /// ```
    pub fn min_dim_with_indices<I: AsIndex>(self, dim: I) -> (Self, Tensor<D, Int>) {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Min Dim With Indices");

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
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor1 = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor2 = Tensor::<2>::from_data([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], &device);
    /// let tensor = tensor1.min_pair(tensor2);
    /// println!("{tensor}");
    /// // [[1.0, -2.0, 3.0], [1.0, 2.0, 3.0]]
    /// ```
    pub fn min_pair(self, other: Self) -> Self {
        let mask = other.clone().lower(self.clone());
        self.mask_where(mask, other)
    }

    /// Clamp element wise between the given min and max values.
    ///
    /// For floating-point tensors, if the result is zero and the input and bounds
    /// include zeros of opposite signs, either sign of zero may be returned.
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
    /// use burn_tensor::{Int, Tensor};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2, Int>::from_ints(
    ///  [
    ///   [1, 2, 3],
    ///   [4, 5, 6],
    ///   [7, 8, 9]
    ///  ],
    ///  &device);
    ///  let tensor = tensor.clamp(2, 6);
    ///  println!("{tensor}");
    ///  // [[2, 2, 3], [4, 5, 6], [6, 6, 6]]
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
    /// use burn_tensor::{Int, Tensor};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2, Int>::from_ints(
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    /// &device);
    /// let tensor = tensor.clamp_min(4);
    /// println!("{tensor}");
    /// // [[4, 4, 4], [4, 5, 6], [7, 8, 9]]
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
    /// use burn_tensor::{Int, Tensor};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2, Int>::from_ints(
    /// [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    /// &device);
    /// let tensor = tensor.clamp_max(5);
    /// println!("{tensor}");
    /// // [[1, 2, 3], [4, 5, 5], [5, 5, 5]]
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
    ///   Negative dimensions are supported and count from the end.
    ///
    /// For floating-point tensors, once a NaN is encountered in a scan, the output at that
    /// position and every later position in the scan is NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[3.0, 5.0, 2.0], [4.0, 1.0, 6.0]], &device);
    /// let result = tensor.clone().cummin(0);
    /// println!("{result}");
    /// // [[3.0, 5.0, 2.0], [3.0, 1.0, 2.0]]
    /// let result = tensor.cummin(1);
    /// println!("{result}");
    /// // [[3.0, 3.0, 2.0], [4.0, 1.0, 1.0]]
    /// ```
    pub fn cummin<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Cummin");
        Self::new(K::cummin(self.primitive, dim))
    }

    /// Computes the cumulative maximum of elements along the given *dimension* or *axis*.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative maximum.
    ///   Negative dimensions are supported and count from the end.
    ///
    /// For floating-point tensors, once a NaN is encountered in a scan, the output at that
    /// position and every later position in the scan is NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[3.0, 1.0, 2.0], [4.0, 5.0, 2.0]], &device);
    /// let result = tensor.clone().cummax(0);
    /// println!("{result}");
    /// // [[3.0, 1.0, 2.0], [4.0, 5.0, 2.0]]
    /// let result = tensor.cummax(1);
    /// println!("{result}");
    /// // [[3.0, 3.0, 3.0], [4.0, 5.0, 5.0]]
    /// ```
    pub fn cummax<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Cummax");
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
    /// For floating-point tensors, a reduced slice produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.max_dim(0);
    /// println!("{tensor}");
    /// // [[5.0, 9.0, 6.0]]
    /// ```
    pub fn max_dim<I: AsIndex>(self, dim: I) -> Self {
        let dim = unwrap_dim_index(dim.try_dim_index(D), "Max Dim");
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
    /// For floating-point tensors, a reduced region produces NaN if it contains a NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    /// let tensor = tensor.max_dims(&[0, 1]);
    /// println!("{tensor}");
    /// // [[9.0]]
    /// ```
    pub fn max_dims<I: AsIndex>(self, dims: &[I]) -> Self {
        dims.iter().fold(self, |tensor, &dim| tensor.max_dim(dim))
    }
}

/// The one-hot class of each index, or `None` when the index is outside
/// `[-num_classes, num_classes)`.
///
/// Indices are converted to integers first, truncating fractional values toward zero as ONNX
/// `OneHot` does; values that can't be converted (NaN, infinities, beyond `i64`) are out of range.
/// The class bounds are then checked on the integers, so they never depend on the index dtype.
fn one_hot_classes(indices: &TensorData, num_classes: usize) -> Vec<Option<usize>> {
    // Every float in [-2^63, 2^63) converts to an i64
    const I64_RANGE: f64 = 9_223_372_036_854_775_808.0;

    let integers: Box<dyn Iterator<Item = Option<i64>>> = if indices.dtype().is_float() {
        // NaN fails the range check; `as` truncates in-range values toward zero
        Box::new(indices.iter::<f64>().map(|value| {
            (-I64_RANGE..I64_RANGE)
                .contains(&value)
                .then_some(value as i64)
        }))
    } else if indices.dtype().is_uint() {
        Box::new(indices.iter::<u64>().map(|value| i64::try_from(value).ok()))
    } else {
        Box::new(indices.iter::<i64>().map(Some))
    };

    let num_classes = i64::try_from(num_classes).unwrap_or(i64::MAX);
    integers
        .map(|index| {
            let index = index?;
            let class = if index < 0 {
                index + num_classes
            } else {
                index
            };
            (0..num_classes).contains(&class).then_some(class as usize)
        })
        .collect()
}
