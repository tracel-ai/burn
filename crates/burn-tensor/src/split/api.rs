
use burn_backend::ops::ComplexTensorOps;
use burn_std::{
    AsIndex, ComplexScalar, DType, Element, ElementConversion, IndexingUpdateOp, ReshapeAnalysis::Split, Scalar, Shape, SliceArg, dtype
};

use crate::{
    Bool, BroadcastArgs, Complex, Device, Float, Int, ReshapeArgs, Tensor, TensorCreationOptions, bool_and_impl, bool_or_impl, check::TensorCheck, kind::{Basic, Numeric}, ops::{BasicOps, BridgeTensor, CompoundTensorKind, FloatMathOps}, split::base::{SplitBackend, SplitTensor}
    //split::base::SplitTensor,
};

impl<const D: usize, K: CompoundTensorKind + Basic> SplitTensor<D, K> {
    fn from_components_array(components: K::ComponentsArray) -> Self {
        Self {
            _kind: core::marker::PhantomData,
            components,
        }
    }
}
//BasicOps
impl<const D: usize> SplitTensor<D, Complex>
{
    /// Select complex tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices of the elements to select.
    pub fn select(self, dim: usize, indices: Tensor<1, Int>) -> Self {
        SplitBackend::complex_select(self.into(), dim, indices.primitive.into()).into()
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_std::backend::Backend;
    /// use burn_std::Tensor;
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
    /// use burn_std::backend::Backend;
    /// use burn_std::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Shape { dims: [2, 3, 4] }
    ///    let shape = tensor.shape();
    /// }
    /// ```
    pub fn shape(&self) -> Shape {
        self.components.as_ref()[0].shape()
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
        indices: Tensor<1, Int>,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        
        match update {
            IndexingUpdateOp::Add => {
                SplitBackend::complex_select_add(self.into(), dim, indices.primitive.into(), values.into()).into()
                
            }
            _ => unimplemented!("Only IndexingUpdateOp::Add is currently implemented for select_assign.")
        }
    }

    /// Transpose the complex tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions.
    pub fn transpose(self) -> Self {
        SplitBackend::complex_transpose(self.into()).into()
    }

    /// Swaps two dimensions of a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitBackend::complex_swap_dims(self.into(), dim1, dim2).into()
    }

    /// Returns the device of the current complex tensor.
    pub fn device(&self) -> Device {
        Float::device(&self.components.as_ref()[0])
    }

    /// Move the complex tensor to the given device.
    pub fn to_device(self, device: &Device) -> Self {
        SplitBackend::complex_to_device(self.into(), device.as_dispatch()).into()
    }

    /// Repeat the complex tensor along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension.
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitBackend::complex_repeat_dim(self.into(), dim, times).into()
    }
    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are equal and `false` elsewhere.
    pub fn equal(self, rhs: Self) -> Tensor<D, Bool> {
        let out_dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_equal(self.into(), rhs.into(), out_dtype).into()))
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are not equal and `false` elsewhere.
    pub fn not_equal(self, rhs: Self) -> Tensor<D, Bool> {
        let out_dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_not_equal(self.into(), rhs.into(), out_dtype).into()))
    }

    /// Tests if any element in the complex tensor evaluates to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if any element is non-zero, `false` otherwise.
    pub fn any(self) -> Tensor<1, Bool> {
        let out_dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_any(self.into(), out_dtype).into()))
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
    pub fn any_dim(self, dim: usize) -> Tensor<D, Bool> {
        let dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_any_dim(self.into(), dim, dtype).into()))
    }


    /// Tests if all elements in the complex tensor evaluate to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if all elements are non-zero, `false` otherwise.
    pub fn all(self) -> Tensor<1, Bool> {
        let dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_all(self.into(), dtype)))
        
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
    pub fn all_dim(self, dim: usize) -> Tensor<D, Bool> {
        let dtype = self.device().settings().bool_dtype;
        Tensor::new(BridgeTensor::bool(SplitBackend::complex_all_dim(self.into(), dim, dtype)))
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
            crate::check!(TensorCheck::permute(fixed_axes));
            SplitBackend::complex_permute(self.into(), &fixed_axes).into()
        }
    }

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
        crate::check!(TensorCheck::flip(D, &transformed_axes));
        SplitBackend::complex_flip(self.into(), &transformed_axes).into()
    }

    /// Returns a tensor containing the elements selected from the given slices.
    ///
    /// This method provides flexible tensor slicing with support for various range types,
    /// negative indices, and stepped slicing. The method accepts both single slices and
    /// arrays of slices, with the [`s!`] macro providing convenient syntax for complex patterns.
    ///
    /// # Arguments
    ///
    /// * `slices` - Can be:
    ///   - A single range for 1D slicing (e.g., `0..5`, `..`, `2..`)
    ///   - An array of ranges (e.g., `[0..2, 1..4]`)
    ///   - The [`s!`] macro output for advanced slicing with steps
    ///   - a `&Vec<Slice>` or `&[Slice]`
    ///
    /// # Behavior
    ///
    /// - Supports partial and full slicing in any number of dimensions
    /// - Handles negative indices by wrapping from the end (-1 is the last element)
    /// - Automatically clamps ranges that exceed tensor dimensions
    /// - Supports stepped slicing for selecting every nth element
    /// - Negative steps reverse the selection order
    ///
    /// # Panics
    ///
    /// - If the number of slices exceeds the tensor's dimensions
    /// - If a range is descending (e.g., 2..1) or empty (e.g., 1..1) without negative step
    /// - If a step is zero
    pub fn slice<S>(self, slices: S) -> Self
    where
        S: SliceArg,
    {
        let shape = self.shape();
        let slices = slices.into_slices(&shape);

        // Validate slices
        crate::check!(TensorCheck::slice::<D>(&shape, &slices));

        // Calculate output shape and check for empty slices
        let mut output_dims = shape.clone();
        for (dim, slice) in slices.iter().enumerate() {
            output_dims[dim] = slice.output_size(shape[dim]);
        }

        // Return empty tensor if any dimension is 0 (empty slice)
        if output_dims.contains(&0) {
            return Self::empty(output_dims, &self.device());
        }
        SplitBackend::complex_slice(self.into(), &slices).into()
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

        crate::check!(TensorCheck::slice_assign::<D>(
            &shape,
            &values.shape(),
            &slices
        ));
        SplitBackend::complex_slice_assign(self.into(), &slices, values.into()).into()
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
    pub fn mask_where(self, mask: Tensor<D, Bool>, source: Self) -> Self {
        SplitBackend::complex_mask_where(self.into(), mask.primitive.into(), source.into()).into()
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
    pub fn gather(self, dim: usize, indices: Tensor<D, Int>) -> Self {
        crate::check!(TensorCheck::gather::<D>(
            dim,
            &self.shape(),
            &indices.shape()
        ));
        let indices = indices.primitive;
        SplitBackend::complex_gather(dim, self.into(),  indices.into()).into()
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
        indices: Tensor<D, Int>,
        values: Self,
        update: burn_std::IndexingUpdateOp,
    ) -> Self {
        crate::check!(TensorCheck::scatter::<D>(
            dim,
            &self.shape(),
            &indices.shape(),
            &values.shape()
        ));
        match update {
            IndexingUpdateOp::Add => {
                SplitBackend::complex_scatter_add(dim, self.into(),  indices.primitive.into(), values.into()).into()
                
            }
            _ => unimplemented!("Only IndexingUpdateOp::Add is currently implemented for scatter.")
        }
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
        options: impl Into<TensorCreationOptions>,
    ) -> Self {
        let options = options.into();
        let device = options.device;
        let fill_value = Scalar::Complex(fill_value.elem::<ComplexScalar<f64>>());
        SplitBackend::complex_full(shape.into(), fill_value, device.as_dispatch(), device.settings().complex_dtype).into()
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
        indices: Tensor<M, Int>,
        values: SplitTensor<DV, Complex>,
        _update: IndexingUpdateOp,
    ) -> Self {
        crate::check!(TensorCheck::scatter_nd::<D, M, DV>(
            &self.shape(),
            &indices.shape(),
            &values.shape()
        ));
        

        SplitBackend::complex_scatter_nd(self.into(), indices.primitive.into(), values.into(), _update).into()
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
        indices: Tensor<M, Int>,
    ) -> SplitTensor<DV, Complex> {
        let _indices = indices.primitive;
        //crate::check!(TensorCheck::gather_nd::<D, M, DV>(&indices.shape()));
        //crate::split_tensor_unary_body!(K,gather_nd,self,indices)
        todo!()
    }
}

fn inner_dtype<K: CompoundTensorKind>(device: &Device) -> DType {
    match K::INNER_KIND_ID {
        crate::ops::TensorKindId::Float => device.settings().float_dtype.into(),
        crate::ops::TensorKindId::Int => device.settings().int_dtype.into(),
        crate::ops::TensorKindId::Bool => device.settings().bool_dtype.into(),
        crate::ops::TensorKindId::Complex => device.settings().complex_dtype.into(),
    }
}
//impl<B, F> Numeric<B> for SplitTensor<F>
impl<const D: usize> SplitTensor<D, Complex>
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
        SplitBackend::complex_add(self.into(), rhs.into()).into()
    }

    /// Applies element-wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to add, element-wise.
    pub fn add_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device =&self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<ComplexScalar<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, device);
        self.add(scalar_tensor)
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
        SplitBackend::complex_sub(self.into(), rhs.into()).into()
    }

    /// Applies element-wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to subtract, element-wise.
    pub fn sub_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<ComplexScalar<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, &device);
        self.sub(scalar_tensor)
    }

    /// Switch sign of each element in the complex tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        SplitBackend::complex_neg(self.into()).into()
    }

    /// Aggregate all elements in the complex tensor with the sum operation.
    pub fn sum(self) -> Self {
        SplitBackend::complex_sum(self.into()).into()
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn sum_dim(self, dim: usize) -> Self {
        SplitBackend::complex_sum_dim(self.into(), dim).into()
    }

    /// Aggregate all elements in the complex tensor with the mean operation.
    pub fn mean(self) -> Self {
        SplitBackend::complex_mean(self.into()).into()
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn mean_dim(self, dim: usize) -> Self {
        SplitBackend::complex_mean_dim(self.into(), dim).into()
    }

    /// Computes the cumulative sum of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    pub fn cumsum(self, dim: usize) -> Self {
        SplitBackend::complex_cumsum(self.into(), dim).into()
    }
}