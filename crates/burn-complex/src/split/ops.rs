mod numeric;
use alloc::vec::Vec;
use burn_std::{AsIndex, DType, Shape, SliceArg};
use burn_tensor::{
    Bool, BroadcastArgs, Complex, Device, Distribution, Element, ElementConversion, Float,
    IndexingUpdateOp, Int, ReshapeArgs, Scalar, Tensor, TensorCreationOptions, TensorData,
    TensorMetadata,
    backend::{Backend, BackendTypes, ExecutionError},
    get_device_settings, try_read_sync,
};

use crate::{
    base::{ComplexTensorBackend, ComplexTensorOps},
    split::{SplitBackend, SplitComplexTensor},
};

impl<B, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    /// Create an empty complex tensor of the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn empty<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        SplitBackend::<B, D>::complex_zeros(shape, &opt.device, dtype.into())
    }

    /// Returns a complex tensor containing the elements selected from the given slices.
    ///
    /// # Arguments
    ///
    /// * `slices` - The slices to select from.
    ///
    /// # Panics
    ///
    /// If the number of slices exceeds the tensor's dimensions.
    pub fn slice<S>(self, slices: S) -> Self
    where
        S: SliceArg,
    {
        let shape = self.shape();
        let slices = slices.into_slices(&shape);

        // Validate slices
        //check!(TensorCheck::slice::<D>(&shape, &slices));

        // Calculate output shape and check for empty slices
        let mut output_dims = shape.clone();
        for (dim, slice) in slices.iter().enumerate() {
            output_dims[dim] = slice.output_size(shape[dim]);
        }

        // Return empty tensor if any dimension is 0 (empty slice)
        if output_dims.contains(&0) {
            return Self::empty(output_dims, &self.device());
        }
        SplitBackend::<B, D>::complex_slice(self, &slices)
    }

    /// Create a complex tensor of the given shape where each element is zero.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn zeros<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let options = options.into();
        let shape = shape.into();
        let device = &options.device;
        let dtype = crate::utils::real_to_complex_dtype(options.resolve_dtype::<Float>());
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B, D>::complex_zeros(shape, device, dtype.into())
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }

    /// Create a complex tensor of the given shape where each element is one.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn ones<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let options = options.into();
        let shape = shape.into();
        let device = &options.device;
        let dtype = crate::utils::real_to_complex_dtype(options.resolve_dtype::<Float>());
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B, D>::complex_ones(shape, device, dtype.into())
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }
}
//BasicOps
impl<B, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    /// Select complex tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices of the elements to select.
    pub fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
        // Uses your existing `select` name.
        SplitBackend::<B, D>::complex_select(self, dim, indices.into_primitive())
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///   let dims = tensor.dims(); // [2, 3, 4]
    ///   println!("{dims:?}");
    /// }
    /// ```
    pub fn dims(&self) -> [usize; D] {
        Self::shape(self).dims()
    }

    /// Returns the shape of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Shape { dims: [2, 3, 4] }
    ///    let shape = tensor.shape();
    /// }
    /// ```
    pub fn shape(&self) -> Shape {
        self.real.shape()
    }

    /// Assign the selected complex tensor elements along the given dimension corresponding to
    /// the given indices from the value tensor to the original tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select.
    /// * `indices` - The indices to select from the tensor.
    /// * `values` - The complex values to assign to the selected indices.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Panics
    ///
    /// If `update` is not `IndexingUpdateOp::Add`. Other operations are currently not implemented.
    pub fn select_assign(
        self,
        dim: usize,
        indices: Tensor<B, 1, Int>,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => SplitBackend::<B, D>::complex_select_add(
                self,
                dim,
                indices.into_primitive(),
                values,
            ),
            _ => unimplemented!(),
        }
    }

    /// Reshape the complex tensor to have the given shape.
    ///
    /// The tensor has the same data and number of elements as the input.
    ///
    /// A `-1` in the shape is used to infer the remaining dimensions, e.g.: `[2, -1]`
    /// will reshape the tensor with [2, 3, 4] dimensions to [2, 12].
    ///
    /// # Arguments
    /// - `shape`: The new shape of the tensor.
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(
        self,
        shape: S,
    ) -> SplitComplexTensor<B, D2> {
        // Convert reshape args to shape
        let shape = shape.into_shape::<D2>(self.shape());
        SplitComplexTensor::new(
            B::float_reshape(self.real, shape.clone()),
            B::float_reshape(self.imag, shape),
        )
    }

    /// Transpose the complex tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions.
    pub fn transpose(self) -> Self {
        SplitBackend::<B, D>::complex_transpose(self)
    }

    /// Swaps two dimensions of a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitBackend::<B, D>::complex_swap_dims(self, dim1, dim2)
    }

    /// Returns the device of the current complex tensor.
    pub fn device(&self) -> B::Device {
        SplitBackend::<B, D>::complex_device(self)
    }

    /// Move the complex tensor to the given device.
    pub fn to_device(self, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_to_device(self, device)
    }

    /// Converts the data of the current complex tensor asynchronously.
    ///
    /// Returns the data as interleaved real and imaginary values.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        SplitBackend::<B, D>::complex_into_interleaved_data(self).await
    }

    /// Create a complex tensor from the given interleaved complex data on the given device.
    ///
    /// # Arguments
    ///
    /// * `data` - The interleaved complex data (alternating real and imaginary values).
    /// * `options` - Options to control creation, including device and dtype.
    pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions<B>>) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        let opt = options.into();
        SplitBackend::<B, D>::complex_from_interleaved_data(
            data.convert::<<SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar>(),
            &opt.device,
        )
    }

    /// Repeat the complex tensor along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension.
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitBackend::<B, D>::complex_repeat_dim(self, dim, times)
    }
    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are equal and `false` elsewhere.
    pub fn equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_equal(self, rhs, out_dtype)
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are not equal and `false` elsewhere.
    pub fn not_equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_not_equal(self, rhs, out_dtype)
    }

    /// Concatenates all complex tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// - If `dim` is higher than the rank.
    /// - If `tensors` is an empty vector.
    /// - If all tensors don't have the same shape (the dimension `dim` is ignored).
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cat(tensors, dim)
    }

    /// Tests if any element in the complex tensor evaluates to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if any element is non-zero, `false` otherwise.
    pub fn any(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_any(self, out_dtype)
    }

    /// Tests if any element in the complex tensor evaluates to non-zero along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input, except in the `dim` axis where
    /// the size is 1, containing `true` if any element along that dimension is non-zero.
    pub fn any_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_any_dim(self, dim, out_dtype)
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if all elements are non-zero, `false` otherwise.
    pub fn all(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_all(self, out_dtype)
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input, except in the `dim` axis where
    /// the size is 1, containing `true` if all elements along that dimension are non-zero.
    pub fn all_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_all_dim(self, dim, out_dtype)
    }

    /// Permute the dimensions of the complex tensor.
    ///
    /// This is a no-op when the resolved `axes` match the current order.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of the dimensions. The length of the axes must equal the
    ///   number of dimensions. The values must be unique and in the range of the number of
    ///   dimensions. Negative values are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    pub fn permute<Dim>(self, axes: [Dim; D]) -> Self
    where
        Dim: AsIndex,
    {
        let mut no_op = true;
        let mut fixed_axes = [0; D];
        for (i, axis) in axes.into_iter().enumerate() {
            let dim = axis.expect_dim_index(D);
            no_op &= dim == i;
            fixed_axes[i] = dim;
        }

        if no_op {
            self
        } else {
            SplitBackend::<B, D>::complex_permute(self, &fixed_axes)
        }
    }

    // pub fn expand(self, shape: Shape) -> Self {
    //     SplitBackend::<B, D>::complex_expand(self, shape)
    // }

    /// Broadcast the complex tensor to the given shape.
    ///
    /// Only singleton dimensions can be expanded to a larger size. Other dimensions must have
    /// the same size (which can be inferred with `-1`).
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape to broadcast the tensor to. Can contain -1 for dimensions that
    ///   should be inferred.
    ///
    /// # Panics
    ///
    /// If the tensor cannot be broadcasted to the given shape.
    pub fn expand<const D2: usize, S: BroadcastArgs<D, D2>>(
        self,
        shape: S,
    ) -> SplitComplexTensor<B, D2> {
        let shape = shape.into_shape(&self.shape());
        // check!(TensorCheck::expand::<D, D2>(
        //     "expand",
        //     &self.shape(),
        //     &shape,
        // ));

        SplitComplexTensor::<B, D2>::new(
            B::float_expand(self.real, shape.clone()),
            B::float_expand(self.imag, shape),
        )
    }

    // pub fn flip(self, axes: &[usize]) -> Self {
    //     SplitBackend::<B, D>::complex_flip(self, axes)
    // }

    /// Reverse the order of elements in the complex tensor along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The dimensions to reverse. The values must be unique and in the range of the
    ///   number of dimensions. Negative values are used as an offset from the end.
    pub fn flip<const N: usize>(self, axes: [isize; N]) -> Self {
        // Convert the axes to usize and handle negative values without using vector
        let mut transformed_axes: [usize; N] = [0; N];
        for (i, &x) in axes.iter().enumerate() {
            transformed_axes[i] = if x < 0 {
                (D as isize + x) as usize
            } else {
                x as usize
            };
        }

        // Check if the axes are valid
        //check!(TensorCheck::flip(D, &transformed_axes));

        SplitComplexTensor::<B, D>::new(
            B::float_flip(self.real, &transformed_axes),
            B::float_flip(self.imag, &transformed_axes),
        )
    }
    /// Returns a view of the complex tensor with an additional dimension of size `size`
    /// obtained by slicing the tensor along `dim` with step `step`.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to unfold.
    /// * `size` - The size of each unfolded window.
    /// * `step` - The step between each window.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[..., windows, ..., size]` where the extra `size` dimension
    /// is appended at the end.
    pub fn unfold<const D2: usize, I: AsIndex>(
        self,
        dim: I,
        size: usize,
        step: usize,
    ) -> SplitComplexTensor<B, D2> {
        let dim = dim.expect_dim_index(D);
        // check!(TensorCheck::unfold::<D, D2>(
        //     "unfold",
        //     &self.shape(),
        //     dim,
        //     size,
        //     step,
        // ));
        SplitComplexTensor::new(
            B::float_unfold(self.real, dim, size, step),
            B::float_unfold(self.imag, dim, size, step),
        )
    }

    /// Assigns values to a slice of the complex tensor and returns the updated tensor.
    ///
    /// # Arguments
    ///
    /// * `slices` - The slice specification indicating where to assign.
    /// * `values` - Tensor with complex values to assign (must match the selected slice dimensions).
    ///
    /// # Panics
    ///
    /// - If slices exceed tensor dimensions.
    /// - If values dimensions don't match the selected slice shape.
    pub fn slice_assign<S>(self, slices: S, values: Self) -> Self
    where
        S: SliceArg,
    {
        let shape = self.shape();
        let slices = slices.into_slices(&shape);

        // Check if any slice produces 0 elements (empty assignment).
        // Empty assignments are no-ops and would cause issues in backend implementations.
        let is_empty_assignment = slices
            .iter()
            .enumerate()
            .any(|(i, slice)| slice.output_size(shape[i]) == 0);

        if is_empty_assignment {
            return self;
        }

        let values_shape = SplitBackend::<B, D>::complex_shape(&values);
        for (i, slice) in slices
            .iter()
            .enumerate()
            .take(slices.len().min(shape.num_dims()))
        {
            let range = slice.to_range(shape[i]);
            assert!(
                range.end <= shape[i],
                "slice_assign: range ({}..{}) exceeds tensor size {} at dim {}",
                range.start,
                range.end,
                shape[i],
                i,
            );
            let expected = range.end - range.start;
            assert_eq!(
                values_shape[i], expected,
                "slice_assign: values shape {} does not match slice length {} at dim {}",
                values_shape[i], expected, i,
            );
        }

        SplitBackend::<B, D>::complex_slice_assign(self, &slices, values)
    }

    /// Update the complex tensor with the value tensor where the mask is true.
    ///
    /// This is similar to [`mask_fill`](Self::mask_fill), however the value is a tensor
    /// instead of a scalar.
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean tensor with the same shape as the input tensor.
    /// * `source` - The complex tensor to use for replacement where the mask is true.
    pub fn mask_where(self, mask: Tensor<B, D, Bool>, source: Self) -> Self {
        SplitBackend::<B, D>::complex_mask_where(self, mask.into_primitive(), source)
    }

    /// Update the complex tensor with the scalar value where the mask is true.
    ///
    /// This is similar to [`mask_where`](Self::mask_where), however the value is a scalar
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean tensor with the same shape as the input tensor.
    /// * `value` - The scalar value to assign where the mask is true.
    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<B, D, Bool>, value: E) -> Self {
        SplitBackend::<B, D>::complex_mask_fill(self, mask.into_primitive(), value.elem())
    }

    /// Gather complex tensor elements corresponding to the given indices from the specified dimension.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i, j, k], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[i, j, k], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[i, j, k]]; // dim = 2`
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn gather(self, dim: usize, indices: Tensor<B, D, Int>) -> Self {
        // check!(TensorCheck::gather::<D>(
        //     dim,
        //     &self.shape(),
        //     &indices.shape()
        // ));
        SplitBackend::<B, D>::complex_gather(dim, self, indices.into_primitive())
    }

    /// Assign the gathered elements corresponding to the given indices along the specified dimension
    /// from the value tensor to the original complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to scatter elements.
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The complex values to scatter into the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    ///
    /// # Panics
    ///
    /// If `update` is not `IndexingUpdateOp::Add`. Other operations are currently not implemented.
    pub fn scatter(
        self,
        dim: usize,
        indices: Tensor<B, D, Int>,
        values: Self,
        update: burn_tensor::IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => SplitBackend::<B, D>::complex_scatter_add(
                dim,
                self,
                indices.into_primitive(),
                values,
            ),
            _ => unimplemented!(),
        }
    }

    /// Applies element-wise equal comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
    }

    /// Applies element-wise non-equality comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_not_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
    }

    /// Create a complex tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The complex value to fill the tensor with.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn full<S: Into<Shape>, E: ElementConversion>(
        shape: S,
        fill_value: E,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Self {
        let opt = options.into();
        let shape = shape.into();

        let e = E::elem::<Complex<f64>>(fill_value);

        let device: &Device<B> = &opt.device;
        //TODO: figure out how to map dtype so that it doesn't just assume Complex<f64>
        SplitComplexTensor::new(
            B::float_from_data(TensorData::full(&shape, e.real()), device),
            B::float_from_data(TensorData::full(shape, e.imag()), device),
        )
    }

    /// Multi-dimensional scatter: update the complex tensor at locations given by `indices`
    /// using the specified `update` operation.
    ///
    /// The size of `indices`'s last axis (call it `K`) indexes the leading `K` dims of `self`;
    /// the batch shape `indices.shape[0..M-1]` is preserved. `values` has shape
    /// `indices.shape[0..M-1] ++ self.shape[K..D]`. Constraints: `K <= D` and `M >= 1`.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The complex values to scatter into the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn scatter_nd<const M: usize, const DV: usize>(
        self,
        indices: Tensor<B, M, Int>,
        values: SplitComplexTensor<B, DV>,
        update: IndexingUpdateOp,
    ) -> Self {
        // check!(TensorCheck::scatter_nd::<D, M, DV>(
        //     &self.shape(),
        //     &indices.shape(),
        //     &values.shape()
        // ));
        let indices = indices.into_primitive();
        let SplitComplexTensor::<B, D> { real, imag, .. } = self;
        let SplitComplexTensor::<B, DV> {
            real: real_values,
            imag: imag_values,
            ..
        } = values;
        SplitComplexTensor::new(
            B::float_scatter_nd(real, indices.clone(), real_values, update),
            B::float_scatter_nd(imag, indices, imag_values, update),
        )
    }

    /// Multi-dimensional gather: collect complex slices from the tensor at multi-index
    /// locations specified by `indices`.
    ///
    /// The size of `indices`'s last axis (call it `K`) indexes the leading `K` dims of `self`;
    /// the batch shape `indices.shape[0..M-1]` is preserved. The output has shape
    /// `indices.shape[0..M-1] ++ self.shape[K..D]`. Constraints: `K <= D` and `M >= 1`.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn gather_nd<const M: usize, const DV: usize>(
        self,
        indices: Tensor<B, M, Int>,
    ) -> SplitComplexTensor<B, DV> {
        let indices = indices.into_primitive();
        let SplitComplexTensor::<B, D> { real, imag, .. } = self;
        //check!(TensorCheck::gather_nd::<D, M, DV>(&indices.shape()));
        SplitComplexTensor::new(
            B::float_gather_nd(real, indices.clone()),
            B::float_gather_nd(imag, indices),
        )
    }
}
//impl<B, F> Numeric<B> for SplitComplexTensor<F>
impl<B: Backend, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    /// Applies element-wise addition operation.
    ///
    /// `y = x2 + x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to add.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_add(self, rhs)
    }

    /// Applies element-wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to add, element-wise.
    pub fn add_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_add(self, scalar_tensor)
    }

    /// Applies element-wise subtraction operation.
    ///
    /// `y = x2 - x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to subtract.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_sub(self, rhs)
    }

    /// Applies element-wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to subtract, element-wise.
    pub fn sub_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_sub(self, scalar_tensor)
    }

    /// Applies element-wise division operation.
    ///
    /// `y = x2 / x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to divide by.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_div(self, rhs)
    }

    /// Applies element-wise division operation with a scalar.
    ///
    /// `y = x / s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to divide by, element-wise.
    pub fn div_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_div(self, scalar_tensor)
    }

    /// Applies element-wise the remainder operation.
    ///
    /// `y = x2 % x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to compute the remainder with.
    pub fn remainder(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_remainder(self, rhs)
    }

    /// Applies element-wise the remainder operation with a scalar.
    ///
    /// `y = x % s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to compute the remainder with, element-wise.
    pub fn remainder_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        SplitBackend::<B, D>::complex_remainder_scalar(self, rhs.elem())
    }

    /// Applies element-wise multiplication operation.
    ///
    /// `y = x2 * x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to multiply.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_mul(self, rhs)
    }

    /// Applies element-wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to multiply, element-wise.
    pub fn mul_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_mul(self, scalar_tensor)
    }

    /// Switch sign of each element in the complex tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        SplitBackend::<B, D>::complex_neg(self)
    }

    /// Returns the signs of the elements of the complex tensor.
    ///
    /// For a non-zero element `z`, returns `z / |z|`. For zero, returns zero.
    pub fn sign(self) -> Self {
        SplitBackend::<B, D>::complex_sign(self)
    }

    /// Aggregate all elements in the complex tensor with the sum operation.
    pub fn sum(self) -> Self {
        SplitBackend::<B, D>::complex_sum(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn sum_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_sum_dim(self, dim)
    }

    /// Aggregate all elements in the complex tensor with the product operation.
    pub fn prod(self) -> Self {
        SplitBackend::<B, D>::complex_prod(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the product operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn prod_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_prod_dim(self, dim)
    }

    /// Aggregate all elements in the complex tensor with the mean operation.
    pub fn mean(self) -> Self {
        SplitBackend::<B, D>::complex_mean(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn mean_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_mean_dim(self, dim)
    }

    /// Computes the cumulative sum of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    pub fn cumsum(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumsum(self, dim)
    }

    /// Computes the cumulative product of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative product.
    pub fn cumprod(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumprod(self, dim)
    }

    /// Applies element-wise power operation with an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The integer tensor to apply the power operation with.
    pub fn powi(self, other: Tensor<B, D, Int>) -> Self {
        SplitBackend::<B, D>::complex_powi(self, other.into_primitive())
    }

    /// Applies element-wise power operation with an integer scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(other, &self.dtype());
        SplitBackend::<B, D>::complex_powi_scalar(self, other)
    }

    /// Create a random complex tensor of the given shape where each element is sampled from
    /// the given distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn random<S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Self {
        // Use the given dtype when provided, otherwise default device dtype
        let opt = options.into();
        let dtype = opt.resolve_dtype::<Float>();
        SplitBackend::<B, D>::complex_random(shape.into(), distribution, &opt.device, dtype.into())
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn into_data(self) -> TensorData {
        self.try_into_data().expect(
            "Error while reading data: use `try_into_data` instead to catch the error at runtime",
        )
    }
    /// Converts the data of the current tensor and returns any error that might have occurred since the
    /// last time the device was synchronized.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn try_into_data(self) -> Result<TensorData, ExecutionError> {
        try_read_sync(self.into_data_async()).expect(
            "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn to_data(&self) -> TensorData {
        self.clone().into_data()
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    pub fn matmul(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_matmul(self, rhs)
    }
}

// ComplexOnlyOps
impl<B, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    /// Returns the complex conjugate of each element.
    ///
    /// For a complex number `a + bi`, the conjugate is `a - bi`.
    pub fn conj(self) -> Self {
        SplitBackend::<B, D>::conj(self)
    }

    /// Returns the argument (phase angle) of each element, in radians.
    ///
    /// For a complex number `a + bi`, the phase is `atan2(b, a)`, ranging from `-π` to `π`.
    pub fn phase(self) -> F {
        SplitBackend::<B, D>::complex_arg(self)
    }

    /// Returns the magnitude (absolute value, modulus) of each element.
    ///
    /// For a complex number `a + bi`, the magnitude is `sqrt(a² + b²)`.
    pub fn magnitude(self) -> F {
        SplitBackend::<B, D>::abs(self)
    }

    /// Applies element-wise complex exponential.
    ///
    /// For a complex number `a + bi`, computes `exp(a) * (cos(b) + i·sin(b))`.
    pub fn exp(self) -> Self {
        SplitBackend::<B, D>::complex_exp(self)
    }

    /// Applies element-wise complex sine.
    pub fn sin(self) -> Self {
        SplitBackend::<B, D>::complex_sin(self)
    }

    /// Create a complex tensor from separate real and imaginary data.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part data.
    /// * `imag` - The imaginary part data.
    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T) -> Self {
        SplitBackend::<B, D>::complex_from_parts(real.into(), imag.into())
    }

    /// Create a complex tensor from interleaved (real, imaginary) data.
    ///
    /// The input data should contain alternating real and imaginary values.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex data.
    /// * `device` - The device to create the tensor on.
    pub fn from_interleaved_data(data: TensorData, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_from_interleaved_data(data, device)
    }

    /// Create a complex tensor from polar form.
    ///
    /// Constructs a complex tensor where each element `z = r · exp(i · θ)`,
    /// given magnitude `r` and phase angle `θ`.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The magnitude (modulus) of each element.
    /// * `phase` - The phase angle of each element, in radians.
    pub fn from_polar(magnitude: F, phase: F) -> Self {
        SplitBackend::<B, D>::complex_from_polar(magnitude, phase)
    }

    /// Applies element-wise complex cosine.
    pub fn cos(self) -> Self {
        SplitBackend::<B, D>::complex_cos(self)
    }

    /// Applies element-wise complex tangent.
    pub fn tan(self) -> Self {
        SplitBackend::<B, D>::complex_tan(self)
    }

    /// Applies element-wise complex arccosine.
    pub fn acos(self) -> Self {
        SplitBackend::<B, D>::complex_acos(self)
    }

    /// Applies element-wise complex hyperbolic arccosine.
    pub fn acosh(self) -> Self {
        SplitBackend::<B, D>::complex_acosh(self)
    }

    /// Applies element-wise complex arcsine.
    pub fn asin(self) -> Self {
        SplitBackend::<B, D>::complex_asin(self)
    }

    /// Applies element-wise complex hyperbolic arcsine.
    pub fn asinh(self) -> Self {
        SplitBackend::<B, D>::complex_asinh(self)
    }

    /// Applies element-wise complex arctangent.
    pub fn atan(self) -> Self {
        SplitBackend::<B, D>::complex_atan(self)
    }

    /// Applies element-wise complex hyperbolic arctangent.
    pub fn atanh(self) -> Self {
        SplitBackend::<B, D>::complex_atanh(self)
    }

    /// Applies element-wise complex natural logarithm.
    ///
    /// For a complex number `z = r · exp(i · θ)`, computes `ln(r) + i · θ`.
    pub fn log(self) -> Self {
        SplitBackend::<B, D>::complex_log(self)
    }
    /// Applies element-wise complex square root.
    pub fn sqrt(self) -> Self {
        SplitBackend::<B, D>::complex_sqrt(self)
    }
}

// /// Module where we defined macros that can be used only in the project.
// pub(crate) mod macros {
//     /// We use a macro for all checks, since the panic message file and line number will match the
//     /// function that does the check instead of a generic error.rs crate private unrelated file
//     /// and line number.
//     macro_rules! check {
//         ($check:expr) => {
//             if let TensorCheck::Failed(check) = $check {
//                 core::panic!("{}", check.format());
//             }
//         };
//     }
//     pub(crate) use check;
// }
