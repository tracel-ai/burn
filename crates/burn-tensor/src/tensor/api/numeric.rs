use crate::alloc::borrow::ToOwned;

use crate::{
    backend::Backend, check, check::TensorCheck, BasicOps, Bool, Distribution, Element,
    ElementConversion, Float, Int, Shape, Tensor, TensorKind,
};
use num_traits::Zero;

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Applies element wise addition operation.
    ///
    /// `y = x2 + x1`
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Add", &self, &other));
        Self::new(K::add(self.primitive, other.primitive))
    }

    /// Applies element wise addition operation with a scalar.
    ///
    /// `y = x + s`
    pub fn add_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::add_scalar(self.primitive, other))
    }

    /// Applies element wise subtraction operation.
    ///
    /// `y = x2 - x1`
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Sub", &self, &other));
        Self::new(K::sub(self.primitive, other.primitive))
    }

    /// Applies element wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    pub fn sub_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::sub_scalar(self.primitive, other))
    }

    /// Applies element wise division operation.
    ///
    /// `y = x2 / x1`
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Div", &self, &other));
        Self::new(K::div(self.primitive, other.primitive))
    }

    /// Applies element wise division operation with a scalar.
    ///
    /// `y = x / s`
    pub fn div_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::div_scalar(self.primitive, other))
    }
    ///
    /// Applies element wise multiplication operation.
    ///
    /// `y = x2 * x1`
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Mul", &self, &other));
        Self::new(K::mul(self.primitive, other.primitive))
    }

    /// Applies element wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    pub fn mul_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::mul_scalar(self.primitive, other))
    }

    /// Switch sign of each element in the tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::new(K::neg(self.primitive))
    }

    /// Returns the signs of the elements of the input tensor.
    pub fn sign(self) -> Self {
        Self::new(K::sign(self.primitive))
    }

    /// Create a tensor of the given shape where each element is zero.
    pub fn zeros<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::zeros(shape.into(), device))
    }

    /// Create a tensor of the given shape where each element is one.
    pub fn ones<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::ones(shape.into(), device))
    }

    /// Create a tensor of the given shape where each element is equal to the provided value.
    pub fn full<S: Into<Shape<D>>, E: ElementConversion>(
        shape: S,
        fill_value: E,
        device: &B::Device,
    ) -> Self {
        Self::new(K::full(shape.into(), fill_value, device))
    }

    /// Aggregate all elements in the tensor with the mean operation.
    pub fn mean(self) -> Tensor<B, 1, K> {
        Tensor::new(K::mean(self.primitive))
    }

    /// Aggregate all elements in the tensor with the sum operation.
    pub fn sum(self) -> Tensor<B, 1, K> {
        Tensor::new(K::sum(self.primitive))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the mean operation.
    pub fn mean_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Mean", dim));
        Self::new(K::mean_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the sum operation.
    pub fn sum_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Sum", dim));
        Self::new(K::sum_dim(self.primitive, dim))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the product operation.
    pub fn prod(self) -> Tensor<B, 1, K> {
        Tensor::new(K::prod(self.primitive))
    }

    /// Aggregate all elements along the given *dimension* or *axis*
    /// in the tensor with the product operation.
    pub fn prod_dim(self, dim: usize) -> Self {
        check!(TensorCheck::aggregate_dim::<D>("Prod", dim));
        Self::new(K::prod_dim(self.primitive, dim))
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        K::equal_elem::<D>(self.primitive, other.elem())
    }

    /// Applies element wise non-equality comparison and returns a boolean tensor.
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        K::not_equal_elem::<D>(self.primitive, other.elem())
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater", &self, &other));
        K::greater(self.primitive, other.primitive)
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn greater_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Greater_equal", &self, &other));
        K::greater_equal(self.primitive, other.primitive)
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower", &self, &other));
        K::lower(self.primitive, other.primitive)
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn lower_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Lower_equal", &self, &other));
        K::lower_equal(self.primitive, other.primitive)
    }

    /// Applies element wise greater comparison and returns a boolean tensor.
    pub fn greater_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::greater_elem(self.primitive, other.elem())
    }

    /// Applies element wise greater-equal comparison and returns a boolean tensor.
    pub fn greater_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::greater_equal_elem(self.primitive, other.elem())
    }

    /// Applies element wise lower comparison and returns a boolean tensor.
    pub fn lower_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::lower_elem(self.primitive, other.elem())
    }

    /// Applies element wise lower-equal comparison and returns a boolean tensor.
    pub fn lower_equal_elem<E: ElementConversion>(self, other: E) -> Tensor<B, D, Bool> {
        K::lower_equal_elem(self.primitive, other.elem())
    }

    /// Update the given tensor with the value tensor where the mask is true.
    ///
    /// This is similar to [mask_fill](Tensor::mask_fill), however the value is a tensor instead of
    /// a scalar.
    pub fn mask_where(self, mask: Tensor<B, D, Bool>, value: Self) -> Self {
        Self::new(K::mask_where(self.primitive, mask, value.primitive))
    }

    /// Update the given tensor with the value where the mask is true.
    ///
    /// This is similar to [mask_where](Tensor::mask_where), however the value is a scalar instead of
    /// a tensor.
    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<B, D, Bool>, value: E) -> Self {
        Self::new(K::mask_fill(self.primitive, mask, value.elem()))
    }

    /// Gather tensor elements corresponding to the given indices from the specified dim.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i, j, k], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[i, j, k], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[i, j, k]]; // dim = 2`
    ///
    /// # Notes
    ///
    /// The index tensor should have the same shape as the original tensor except for the dim
    /// specified.
    pub fn gather(self, dim: usize, indices: Tensor<B, D, Int>) -> Self {
        check!(TensorCheck::gather::<D>(
            dim,
            &self.shape(),
            &indices.shape()
        ));

        Self::new(K::gather(dim, self.primitive, indices))
    }

    /// Assign the gathered elements corresponding to the given indices along the specified dimension
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i, j, k], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[i, j, k], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[i, j, k]] += values[i, j, k]; // dim = 2`
    ///
    /// # Notes
    ///
    /// The index tensor should have the same shape as the original tensor except for the specified
    /// dimension. The value and index tensors should have the same shape.
    ///
    /// Other references to the input tensor will not be modified by this operation.
    pub fn scatter(self, dim: usize, indices: Tensor<B, D, Int>, values: Self) -> Self {
        check!(TensorCheck::scatter::<D>(
            dim,
            &self.shape(),
            &indices.shape(),
            &values.shape()
        ));

        Self::new(K::scatter(dim, self.primitive, indices, values.primitive))
    }

    /// Select the tensor elements along the given dimension corresponding to the given indices.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[j], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[k]]; // dim = 2`
    pub fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
        check!(TensorCheck::select::<D>(dim));
        Self::new(K::select(self.primitive, dim, indices))
    }

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[j], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[k]] += values[i, j, k]; // dim = 2`
    pub fn select_assign(
        self,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Tensor<B, D, K>,
    ) -> Self {
        check!(TensorCheck::select_assign::<D>(dim));

        Self::new(K::select_assign(
            self.primitive,
            dim,
            indices,
            values.primitive,
        ))
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
    pub fn max(self) -> Tensor<B, 1, K> {
        Tensor::new(K::max(self.primitive))
    }

    /// Find the maximum value along the given dimension.
    pub fn max_dim(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));

        Tensor::new(K::max_dim(self.primitive, dim))
    }

    /// Find the maximum value along the given dimension.
    ///
    /// Also returns the indices.
    pub fn max_dim_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::aggregate_dim::<D>("Max", dim));

        let (tensor, index) = K::max_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Finds the maximum pair wise values with another Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find maximum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the maximum value found
    /// in the input tensors.
    pub fn max_pair(self, other: Self) -> Self {
        let mask = self.clone().lower(other.clone());
        self.mask_where(mask, other)
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
    pub fn min(self) -> Tensor<B, 1, K> {
        Tensor::new(K::min(self.primitive))
    }

    /// Find the minimum value along the given dimension.
    pub fn min_dim(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));
        Tensor::new(K::min_dim(self.primitive, dim))
    }

    /// Find the minimum value along the given dimension.
    ///
    /// Also returns the indices.
    pub fn min_dim_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::aggregate_dim::<D>("Min", dim));

        let (tensor, index) = K::min_dim_with_indices(self.primitive, dim);

        let tensor = Tensor::new(tensor);
        let index = Tensor::new(index);

        (tensor, index)
    }

    /// Finds the minimum pair wise values with another Tensor
    ///
    /// # Arguments
    ///
    /// * `other` - Other tensor to find minimum elements with
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors containing the minimum value found
    /// between each element of the two source tensors.
    pub fn min_pair(self, other: Self) -> Self {
        let mask = other.clone().lower(self.clone());
        self.mask_where(mask, other)
    }

    /// Clamp the tensor between the given min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped between the given min and max values.
    pub fn clamp<E: ElementConversion>(self, min: E, max: E) -> Self {
        Self::new(K::clamp(self.primitive, min.elem(), max.elem()))
    }

    /// Clamps a tensor under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped under the given min value.
    pub fn clamp_min<E: ElementConversion>(self, min: E) -> Self {
        Self::new(K::clamp_min(self.primitive, min.elem()))
    }

    /// Clamps a tensor over a maximum value.
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
    pub fn clamp_max<E: ElementConversion>(self, max: E) -> Self {
        Self::new(K::clamp_max(self.primitive, max.elem()))
    }

    /// Apply element wise absolute value operation
    pub fn abs(self) -> Self {
        Self::new(K::abs(self.primitive))
    }

    /// Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input,
    /// the other elements of the result tensor out are set to 0.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///        [
    ///          [1, 2, 3],
    ///          [4, 5, 6],
    ///          [7, 8, 9]
    ///        ],
    ///        &device
    ///    );
    ///    let tensor = tensor.triu(1);
    ///    println!("{}", tensor);
    ///    // Tensor { data: [
    ///    //   [0, 2, 3],
    ///    //   [0, 0, 6],
    ///    //   [0, 0, 0]
    ///    // ], ... }
    /// }
    /// ```
    pub fn triu(self, diagonal: i64) -> Self {
        check!(TensorCheck::tri::<{ D }>());

        // last two dimensions
        let shape = &self.shape().dims[D - 2..].to_owned();

        let mask = Tensor::<B, 2, Bool>::triu_mask(shape, diagonal, &self.device()).unsqueeze();
        self.mask_fill(mask, 0)
    }

    /// Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input,
    /// the other elements of the result tensor out are set to 0.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Int, Tensor};
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 2, Int>::from_ints(
    ///        [
    ///          [1, 2, 3],
    ///          [4, 5, 6],
    ///          [7, 8, 9]
    ///        ],
    ///        &device
    ///    );
    ///
    ///    let tensor = tensor.tril(-1);
    ///    println!("{}", tensor);
    ///    // Tensor { data: [
    ///    //   [0, 0, 0],
    ///    //   [4, 0, 0],
    ///    //   [7, 8, 0]
    ///    // ], ... }
    /// }
    /// ```
    pub fn tril(self, diagonal: i64) -> Self {
        check!(TensorCheck::tri::<{ D }>());

        // last two dimensions
        let shape = &self.shape().dims[D - 2..].to_owned();

        let mask = Tensor::<B, 2, Bool>::tril_mask(shape, diagonal, &self.device()).unsqueeze();
        self.mask_fill(mask, 0)
    }

    /// Applies element wise power operation with a float Tensor
    pub fn powf(self, other: Self) -> Self {
        Self::new(K::powf(self.primitive, other.primitive))
    }

    /// Applies element wise power operation with a float scalar
    pub fn powf_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::powf_scalar(self.primitive, other))
    }

    /// Applies element wise power operation with a integer Tensor
    pub fn powi(self, other: Self) -> Self {
        Self::new(K::powi(self.primitive, other.primitive))
    }

    /// Applies element wise power operation with a integer scalar
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        Self::new(K::powi_scalar(self.primitive, other))
    }

    /// Checks element wise if the tensor is close to another tensor.
    ///
    /// The tolerance is defined by the following equation:
    ///
    /// ```text
    /// abs(a - b) <= (atol + rtol * abs(b))
    ///
    /// where `a` is the first tensor, `b` is the second tensor, `rtol` is the relative tolerance,
    /// and `atol` is the absolute tolerance.
    /// ```
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with.
    /// * `rtol` - Optional relative tolerance. Default is 1e-5.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors.
    pub fn is_close(self, other: Self, rtol: Option<f64>, atol: Option<f64>) -> Tensor<B, D, Bool> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);

        K::lower_equal(
            K::abs(K::sub(self.primitive, other.primitive.clone())),
            K::add_scalar(K::mul_scalar(K::abs(other.primitive), rtol), atol),
        )
    }

    /// Checks if all elements are close to another tensor.
    ///
    /// The tolerance is defined by the following equation:
    ///
    /// ```text
    ///
    /// abs(a - b) <= (atol + rtol * abs(b))
    ///
    /// where `a` is the first tensor, `b` is the second tensor, `rtol` is the relative tolerance,
    /// and `atol` is the absolute tolerance.
    ///
    /// ```
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with.
    /// * `rtol` - Optional relative tolerance. Default is 1e-5.
    /// * `atol` - Optional absolute tolerance. Default is 1e-8.
    ///
    /// # Returns
    ///
    /// A boolean scalar.
    ///
    /// # Remarks
    ///
    /// This method is only available for non-wasm targets or when the `wasm-sync` feature is enabled.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn all_close(self, other: Self, rtol: Option<f64>, atol: Option<f64>) -> bool {
        self.is_close(other, rtol, atol).all().into_scalar()
    }

    /// Converts the tensor to a boolean tensor by checking if the elements are non-zero.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor.
    pub fn bool(self) -> Tensor<B, D, Bool> {
        K::not_equal_elem::<D>(self.primitive, K::Elem::zero())
    }

    /// Create a random tensor of the given shape on the given device where each element is
    /// sampled from the given distribution.
    pub fn random<S: Into<Shape<D>>>(
        shape: S,
        distribution: Distribution,
        device: &B::Device,
    ) -> Self {
        Self::new(K::random(shape.into(), distribution, device))
    }

    /// Sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn sort(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::sort_dim::<D>("Sort", dim));
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ false))
    }

    /// Sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn sort_descending(self, dim: usize) -> Tensor<B, D, K> {
        check!(TensorCheck::sort_dim::<D>("Sort", dim));
        Tensor::new(K::sort(self.primitive, dim, /*descending*/ true))
    }

    /// Sort the elements by value in ascending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn sort_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) =
            K::sort_with_indices(self.primitive, dim, /*descending*/ false);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Sort the elements by value in descending order along a given dimension.
    /// Also returns the indices.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn sort_descending_with_indices(self, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        check!(TensorCheck::sort_dim::<D>("Sort_with_indices", dim));
        let (values, indices) = K::sort_with_indices(self.primitive, dim, /*descending*/ true);
        (Tensor::new(values), Tensor::new(indices))
    }

    /// Returns the indices that sort the elements by value in ascending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn argsort(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ false))
    }

    /// Returns the indices that sort the elements by value in descending order along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn argsort_descending(self, dim: usize) -> Tensor<B, D, Int> {
        check!(TensorCheck::sort_dim::<D>("Argsort", dim));
        Tensor::new(K::argsort(self.primitive, dim, /*descending*/ true))
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn topk(self, k: usize, dim: usize) -> Tensor<B, D, K> {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        self.sort_descending(dim).select(dim, k_indices)
    }

    /// Returns the `k` largest elements of the given input tensor along a given dimension.
    /// Also returns the indices.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    pub fn topk_with_indices(self, k: usize, dim: usize) -> (Tensor<B, D, K>, Tensor<B, D, Int>) {
        let k_indices = Tensor::arange(0..k as i64, &self.device());
        let (values, indices) = self.sort_descending_with_indices(dim);
        (
            values.select(dim, k_indices.clone()),
            indices.select(dim, k_indices),
        )
    }
}

impl<B, K> Tensor<B, 2, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    /// Creates a new 2D tensor with ones on the diagonal and zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the square matrix.
    pub fn eye(size: usize, device: &B::Device) -> Self {
        let indices = Tensor::<B, 1, Int>::arange(0..size as i64, device).unsqueeze();
        let ones = K::ones([1, size].into(), device);
        let zeros = K::zeros([size, size].into(), device);
        Self::new(K::scatter(0, zeros, indices, ones))
    }
}

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait Numeric<B: Backend>: BasicOps<B>
where
    Self::Elem: Element,
{
    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The sum of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding tensors, users should prefer the [Tensor::add](Tensor::add) function,
    /// which is more high-level and designed for public use.
    fn add<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The sum of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding a scalar to a tensor, users should prefer the [Tensor::add_scalar](Tensor::add_scalar) function,
    /// which is more high-level and designed for public use.
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Subtracts two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The difference of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting tensors, users should prefer the [Tensor::sub](Tensor::sub) function,
    /// which is more high-level and designed for public use.
    fn sub<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The difference of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting a scalar from a tensor, users should prefer the [Tensor::sub_scalar](Tensor::sub_scalar) function,
    /// which is more high-level and designed for public use.
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Divides two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The quotient of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing tensors, users should prefer the [Tensor::div](Tensor::div) function,
    /// which is more high-level and designed for public use.
    fn div<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Divides a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The quotient of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing a tensor by a scalar, users should prefer the [Tensor::div_scalar](Tensor::div_scalar) function,
    /// which is more high-level and designed for public use.
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The product of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying tensors, users should prefer the [Tensor::mul](Tensor::mul) function,
    /// which is more high-level and designed for public use.
    fn mul<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Multiplies a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The product of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying a tensor by a scalar, users should prefer the [Tensor::mul_scalar](Tensor::mul_scalar) function,
    /// which is more high-level and designed for public use.
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Negates a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to negate.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For negating a tensor, users should prefer the [Tensor::neg](Tensor::neg) function,
    /// which is more high-level and designed for public use.
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Returns the signs of the elements of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The signs of the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the signs of the elements of a tensor, users should prefer the [Tensor::sign](Tensor::sign) function,
    /// which is more high-level and designed for public use.
    fn sign<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with zeros.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with zeros, users should prefer the [Tensor::zeros](Tensor::zeros) function,
    /// which is more high-level and designed for public use.
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with ones.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with ones, users should prefer the [Tensor::ones](Tensor::ones) function,
    /// which is more high-level and designed for public use.
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;

    /// Creates a tensor filled with elements equal to the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with elements equal to the given value
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with a specific value, users should prefer the [Tensor::full](Tensor::full) function,
    /// which is more high-level and designed for public use.
    fn full<const D: usize, E: ElementConversion>(
        shape: Shape<D>,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive<D>;

    /// Sums all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor, users should prefer the [Tensor::sum](Tensor::sum) function,
    /// which is more high-level and designed for public use.
    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;

    /// Sums all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor along a dimension, users should prefer the [Tensor::sum_dim](Tensor::sum_dim) function,
    /// which is more high-level and designed for public use.
    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;

    /// Computes the product of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor, users should prefer the
    /// [Tensor::prod](Tensor::prod) function,
    /// which is more high-level and designed for public use.
    fn prod<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;

    /// Computes the product of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    /// * `dim` - The dimension along which to compute the product.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor along a dimension, users should
    /// prefer the [Tensor::prod_dim](Tensor::prod_dim) function,
    /// which is more high-level and designed for public use.
    ///
    ///
    fn prod_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;

    /// Computes the mean of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor, users should prefer the [Tensor::mean](Tensor::mean) function,
    /// which is more high-level and designed for public use.
    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;

    /// Computes the mean of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    /// * `dim` - The dimension along which to compute the mean.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor along a dimension, users should prefer
    /// the [Tensor::mean_dim](Tensor::mean_dim) function, which is more high-level and designed for public use.
    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;

    /// Element-wise equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise equality between two tensors, users should prefer the [Tensor::equal_elem](Tensor::equal_elem)
    /// function, which is more high-level and designed for public use.
    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool>;

    /// Element-wise non-equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise non-equality between two tensors, users should prefer the [Tensor::not_equal_elem](Tensor::not_equal_elem)
    /// function, which is more high-level and designed for public use.
    fn not_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise greater than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the corresponding element
    /// of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between two tensors, users should prefer the [Tensor::greater](Tensor::greater) function,
    /// which is more high-level and designed for public use.
    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise greater than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the right hand side
    /// scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::greater_elem](Tensor::greater_elem) function, which is more high-level and designed for public use.
    fn greater_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem)
        -> Tensor<B, D, Bool>;

    /// Element-wise greater than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the
    /// corresponding element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between two tensors, users should prefer
    /// the [Tensor::greater_equal](Tensor::greater_equal) function, which is more high-level and designed for public use.
    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise greater than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the right
    /// hand side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::greater_equal_elem](Tensor::greater_equal_elem) function, which is more high-level and designed for public use.
    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise less than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the corresponding element of
    /// the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between two tensors, users should prefer the [Tensor::lower](Tensor::lower) function,
    /// which is more high-level and designed for public use.
    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise less than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the right hand side scalar,
    /// and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::lower_elem](Tensor::lower_elem) function, which is more high-level and designed for public use.
    fn lower_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool>;

    /// Element-wise less than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the corresponding
    /// element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between two tensors, users should prefer
    /// the [Tensor::lower_equal](Tensor::lower_equal) function, which is more high-level and designed for public use.
    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Element-wise less than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the right hand
    /// side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between a tensor and a scalar, users should prefer
    /// the [Tensor::lower_equal_elem](Tensor::lower_equal_elem) function, which is more high-level and designed for public use.
    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool>;

    /// Selects elements from a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from if the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for selecting elements.
    /// * `source` - The tensor to select elements from when the corresponding element of the mask is false.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element of the left hand side tensor if the corresponding element of the mask
    /// is true, and from the corresponding element of the right hand side tensor otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor based on a boolean mask, users should prefer the
    /// [Tensor::mask_where](Tensor::mask_where) function, which is more high-level and designed for public use.
    fn mask_where<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        source: Self::Primitive<D>,
    ) -> Self::Primitive<D>;

    /// Fills elements of a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor where will be overwritten with the value
    ///              when the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for filling elements.
    /// * `value` - The value to fill elements with when the corresponding element of the mask is true.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element unmodified if the corresponding element of the mask is false, and
    /// filled with the value otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For filling elements of a tensor based on a boolean mask, users should prefer the
    /// [Tensor::mask_fill](Tensor::mask_fill) function, which is more high-level and designed for public use.
    fn mask_fill<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        value: Self::Elem,
    ) -> Self::Primitive<D>;

    /// Gathers elements from a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to gather elements.
    /// * `tensor` - The tensor to gather elements from.
    /// * `indices` - The indices of the elements to gather.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For gathering elements from a tensor along an axis, users should prefer the
    /// [Tensor::gather](Tensor::gather) function, which is more high-level and designed for public use.
    fn gather<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
    ) -> Self::Primitive<D>;

    /// Scatters elements into a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to scatter elements.
    /// * `tensor` - The tensor to scatter elements into.
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The values to scatter into the tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For scattering elements into a tensor along an axis, users should prefer the [Tensor::scatter](Tensor::scatter) function,
    /// which is more high-level and designed for public use.
    fn scatter<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D>;

    /// Select tensor elements along the given dimension corresponding for the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from.
    /// * `dim` - The axis along which to select elements.
    /// * `indices` - The indices of the elements to select.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor along an axis, users should prefer the
    /// [Tensor::select](Tensor::select) function, which is more high-level and designed for public use.
    fn select<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D>;

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to assign elements to.
    /// * `dim` - The axis along which to assign elements.
    /// * `indices` - The indices of the elements to assign.
    /// * `values` - The values to assign to the tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For assigning elements to a tensor along an axis, users should prefer the
    /// [Tensor::select_assign](Tensor::select_assign) function, which is more high-level and designed for public use.
    fn select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D>;

    /// Gets the indices of the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the maximum elements.
    /// * `tensor` - The tensor to get the indices of the maximum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// maximum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::argmax](Tensor::argmax) function, which is more high-level and designed for public use.
    fn argmax<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> B::IntTensorPrimitive<D>;

    /// Gets the indices of the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the minimum elements.
    /// * `tensor` - The tensor to get the indices of the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// minimum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::argmin](Tensor::argmin) function, which is more high-level and designed for public use.
    fn argmin<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> B::IntTensorPrimitive<D>;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the maximum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max](Tensor::max) function, which is more high-level and designed for public use.
    fn max<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the maximum element
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max_dim](Tensor::max_dim) function, which is more high-level and designed for public use.
    fn max_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tuple containing the maximum element of the input tensor, and a tensor with the same shape
    /// as the input tensor, where each element is the index of the maximum element of the input tensor
    /// at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    /// [Tensor::max_dim_with_indices](Tensor::max_dim_with_indices) function, which is more high-level and designed for public use.
    fn max_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, B::IntTensorPrimitive<D>);

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the minimum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min](Tensor::min) function, which is more high-level and designed for public use.
    fn min<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1>;

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    /// * `dim` - The axis along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the minimum element
    /// of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min_dim](Tensor::min_dim) function, which is more high-level and designed for public use.
    fn min_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D>;

    /// Gets the minimum elements and indices of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// each element is the minimum element of the input tensor at the corresponding index
    /// along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    /// [Tensor::min_dim_with_indices](Tensor::min_dim_with_indices) function, which is more high-level and designed for public use.
    fn min_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, B::IntTensorPrimitive<D>);

    /// Clamp the tensor between the given min and max values.
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
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor between the given min and max values, users should prefer the
    /// [Tensor::clamp](Tensor::clamp) function, which is more high-level and designed for public use.
    fn clamp<const D: usize>(
        tensor: Self::Primitive<D>,
        min: Self::Elem,
        max: Self::Elem,
    ) -> Self::Primitive<D>;

    /// Clamps a tensor under a minimum value.
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
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor under a minimum value, users should prefer the
    /// [Tensor::clamp_min](Tensor::clamp_min) function, which is more high-level and designed for public use.
    fn clamp_min<const D: usize>(tensor: Self::Primitive<D>, min: Self::Elem)
        -> Self::Primitive<D>;

    /// Clamps a tensor over a maximum value.
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
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor over a maximum value, users should prefer the
    /// [Tensor::clamp_max](Tensor::clamp_max) function, which is more high-level and designed for public use.
    fn clamp_max<const D: usize>(tensor: Self::Primitive<D>, max: Self::Elem)
        -> Self::Primitive<D>;

    /// Calculate absolute value on all elements of a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to apply abs to.
    ///
    /// # Returns
    ///
    /// A tensor with absolute values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For calculating abs of the elements of a tensor, users should prefer the [Tensor::abs](Tensor::abs) function,
    /// which is more high-level and designed for public use.
    fn abs<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D>;

    /// Element-wise power of a tensor to a float tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>)
        -> Self::Primitive<D>;

    /// Element-wise power of a tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Primitive<D>)
        -> Self::Primitive<D>;

    /// Element-wise power of a tensor to a scalar float
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Element-wise power of a tensor to a scalar int
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D>;

    /// Create a random tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the output tensor.
    /// * `distribution` - The distribution used to sample.
    /// * `device` - The device to use.
    ///
    /// # Returns
    ///
    /// A new tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::random](Tensor::random) function,
    /// which is more high-level and designed for public use.
    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &B::Device,
    ) -> Self::Primitive<D>;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where the elements are sorted by value.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::sort](Tensor::sort) function,
    /// which is more high-level and designed for public use.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> Self::Primitive<D>;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// the elements are sorted by value and the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For sorting the elements of a tensor, users should prefer the
    /// [Tensor::sort_with_indices](Tensor::sort_with_indices) function, which is more high-level
    /// and designed for public use.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive<D>, <Int as TensorKind<B>>::Primitive<D>);

    /// Returns the indices that sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::argsort](Tensor::argsort) function,
    /// which is more high-level and designed for public use.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn argsort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive<D>;
}

impl<B: Backend> Numeric<B> for Int {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_add(lhs, rhs)
    }
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_add_scalar(lhs, rhs.elem())
    }
    fn sub<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_sub(lhs, rhs)
    }
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_sub_scalar(lhs, rhs.elem())
    }
    fn div<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_div(lhs, rhs)
    }
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_div_scalar(lhs, rhs.elem())
    }
    fn mul<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_mul(lhs, rhs)
    }
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_mul_scalar(lhs, rhs.elem())
    }
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::int_neg(tensor)
    }
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_zeros(shape, device)
    }
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_ones(shape, device)
    }
    fn full<const D: usize, E: ElementConversion>(
        shape: Shape<D>,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::int_full(shape, fill_value.elem(), device)
    }

    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_sum_dim(tensor, dim)
    }

    fn prod<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_prod(tensor)
    }

    fn prod_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_prod_dim(tensor, dim)
    }

    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_mean(tensor)
    }
    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_mean_dim(tensor, dim)
    }

    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_equal_elem(lhs, rhs))
    }
    fn not_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_not_equal_elem(lhs, rhs))
    }
    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater(lhs, rhs))
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_elem(lhs, rhs))
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_equal(lhs, rhs))
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_greater_equal_elem(lhs, rhs))
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower(lhs, rhs))
    }

    fn lower_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_elem(lhs, rhs))
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_equal(lhs, rhs))
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_lower_equal_elem(lhs, rhs))
    }

    fn mask_where<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        source: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_mask_where(tensor, mask.primitive, source)
    }

    fn mask_fill<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        value: Self::Elem,
    ) -> Self::Primitive<D> {
        B::int_mask_fill(tensor, mask.primitive, value)
    }

    fn select<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D> {
        B::int_select(tensor, dim, indices.primitive)
    }

    fn select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_select_assign(tensor, dim, indices.primitive, values)
    }
    fn gather<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
    ) -> Self::Primitive<D> {
        B::int_gather(dim, tensor, indices.primitive)
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_scatter(dim, tensor, indices.primitive, values)
    }

    fn argmax<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        B::int_argmax(tensor, dim)
    }

    fn argmin<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        B::int_argmin(tensor, dim)
    }

    fn max<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_max(tensor)
    }

    fn max_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_max_dim(tensor, dim)
    }

    fn max_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn min<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::int_min(tensor)
    }

    fn min_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::int_min_dim(tensor, dim)
    }

    fn min_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        B::int_min_dim_with_indices(tensor, dim)
    }

    fn clamp<const D: usize>(
        tensor: Self::Primitive<D>,
        min: B::IntElem,
        max: B::IntElem,
    ) -> Self::Primitive<D> {
        B::int_clamp(tensor, min, max)
    }

    fn clamp_min<const D: usize>(
        tensor: Self::Primitive<D>,
        min: B::IntElem,
    ) -> Self::Primitive<D> {
        B::int_clamp_min(tensor, min)
    }

    fn clamp_max<const D: usize>(
        tensor: Self::Primitive<D>,
        max: B::IntElem,
    ) -> Self::Primitive<D> {
        B::int_clamp_max(tensor, max)
    }

    fn abs<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::int_abs(tensor)
    }

    fn powf<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_powf(lhs, B::int_into_float(rhs))
    }

    fn powf_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_powf_scalar(lhs, rhs.elem())
    }

    fn powi<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::int_powi(lhs, rhs)
    }

    fn powi_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::int_powf_scalar(lhs, rhs.elem())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::int_random(shape, distribution, device)
    }

    fn sign<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::int_sign(tensor)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> Self::Primitive<D> {
        B::int_sort(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive<D>, <Int as TensorKind<B>>::Primitive<D>) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn argsort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::int_argsort(tensor, dim, descending)
    }
}

impl<B: Backend> Numeric<B> for Float {
    fn add<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::float_add(lhs, rhs)
    }
    fn add_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_add_scalar(lhs, rhs.elem())
    }
    fn sub<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::float_sub(lhs, rhs)
    }
    fn sub_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_sub_scalar(lhs, rhs.elem())
    }
    fn div<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::float_div(lhs, rhs)
    }
    fn div_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_div_scalar(lhs, rhs.elem())
    }
    fn mul<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> <Float as TensorKind<B>>::Primitive<D> {
        B::float_mul(lhs, rhs)
    }
    fn mul_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_mul_scalar(lhs, rhs.elem())
    }
    fn neg<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::float_neg(tensor)
    }
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::float_zeros(shape, device)
    }
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::float_ones(shape, device)
    }

    fn full<const D: usize, E: ElementConversion>(
        shape: Shape<D>,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::float_full(shape, fill_value.elem(), device)
    }

    fn sum<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::float_sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::float_sum_dim(tensor, dim)
    }

    fn prod<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::float_prod(tensor)
    }

    fn prod_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::float_prod_dim(tensor, dim)
    }

    fn mean<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::float_mean(tensor)
    }

    fn mean_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::float_mean_dim(tensor, dim)
    }

    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_equal_elem(lhs, rhs))
    }
    fn not_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_not_equal_elem(lhs, rhs))
    }
    fn greater<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_greater(lhs, rhs))
    }

    fn greater_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_greater_elem(lhs, rhs))
    }

    fn greater_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_greater_equal(lhs, rhs))
    }

    fn greater_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_greater_equal_elem(lhs, rhs))
    }

    fn lower<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_lower(lhs, rhs))
    }

    fn lower_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_lower_elem(lhs, rhs))
    }

    fn lower_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_lower_equal(lhs, rhs))
    }

    fn lower_equal_elem<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Elem,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::float_lower_equal_elem(lhs, rhs))
    }

    fn mask_where<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        source: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::float_mask_where(tensor, mask.primitive, source)
    }

    fn mask_fill<const D: usize>(
        tensor: Self::Primitive<D>,
        mask: Tensor<B, D, Bool>,
        value: Self::Elem,
    ) -> Self::Primitive<D> {
        B::float_mask_fill(tensor, mask.primitive, value)
    }

    fn select<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
    ) -> Self::Primitive<D> {
        B::float_select(tensor, dim, indices.primitive)
    }

    fn select_assign<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::float_select_assign(tensor, dim, indices.primitive, values)
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
    ) -> Self::Primitive<D> {
        B::float_gather(dim, tensor, indices.primitive)
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: Self::Primitive<D>,
        indices: Tensor<B, D, Int>,
        values: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::float_scatter(dim, tensor, indices.primitive, values)
    }

    fn argmax<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        B::float_argmax(tensor, dim)
    }

    fn argmin<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> <B as Backend>::IntTensorPrimitive<D> {
        B::float_argmin(tensor, dim)
    }

    fn max<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::float_max(tensor)
    }

    fn max_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::float_max_dim(tensor, dim)
    }

    fn max_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        B::float_max_dim_with_indices(tensor, dim)
    }

    fn min<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<1> {
        B::float_min(tensor)
    }

    fn min_dim<const D: usize>(tensor: Self::Primitive<D>, dim: usize) -> Self::Primitive<D> {
        B::float_min_dim(tensor, dim)
    }

    fn min_dim_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> (Self::Primitive<D>, <B as Backend>::IntTensorPrimitive<D>) {
        B::float_min_dim_with_indices(tensor, dim)
    }

    fn clamp<const D: usize>(
        tensor: Self::Primitive<D>,
        min: B::FloatElem,
        max: B::FloatElem,
    ) -> Self::Primitive<D> {
        B::float_clamp(tensor, min, max)
    }

    fn clamp_min<const D: usize>(
        tensor: Self::Primitive<D>,
        min: B::FloatElem,
    ) -> Self::Primitive<D> {
        B::float_clamp_min(tensor, min)
    }

    fn clamp_max<const D: usize>(
        tensor: Self::Primitive<D>,
        max: B::FloatElem,
    ) -> Self::Primitive<D> {
        B::float_clamp_max(tensor, max)
    }

    fn abs<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::float_abs(tensor)
    }

    fn powf<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::float_powf(lhs, rhs)
    }

    fn powf_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_powf_scalar(lhs, rhs.elem())
    }

    fn powi<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Self::Primitive<D> {
        B::float_powf(lhs, rhs)
    }

    fn powi_scalar<const D: usize, E: ElementConversion>(
        lhs: Self::Primitive<D>,
        rhs: E,
    ) -> Self::Primitive<D> {
        B::float_powf_scalar(lhs, rhs.elem())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::float_random(shape, distribution, device)
    }

    fn sign<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        B::float_sign(tensor)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> Self::Primitive<D> {
        B::float_sort(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn sort_with_indices<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive<D>, <Int as TensorKind<B>>::Primitive<D>) {
        B::float_sort_with_indices(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn argsort<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        descending: bool,
    ) -> <Int as TensorKind<B>>::Primitive<D> {
        B::float_argsort(tensor, dim, descending)
    }
}

impl<B, const D: usize, K> core::ops::Add<Self> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn add(self, rhs: Tensor<B, D, K>) -> Self {
        Self::add(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Add<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn add(self, other: E) -> Self {
        Tensor::add_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Sub<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn sub(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::sub(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Sub<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn sub(self, other: E) -> Self {
        Tensor::sub_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Div<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn div(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::div(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Div<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn div(self, other: E) -> Self {
        Tensor::div_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Mul<Tensor<B, D, K>> for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn mul(self, rhs: Tensor<B, D, K>) -> Self {
        Tensor::mul(self, rhs)
    }
}

impl<E, const D: usize, B, K> core::ops::Mul<E> for Tensor<B, D, K>
where
    E: ElementConversion,
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn mul(self, other: E) -> Self {
        Tensor::mul_scalar(self, other)
    }
}

impl<B, const D: usize, K> core::ops::Neg for Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    type Output = Self;

    fn neg(self) -> Self {
        Tensor::neg(self)
    }
}
