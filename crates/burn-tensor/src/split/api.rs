use std::ptr;

use burn_std::{
    AsIndex, ComplexScalar, DType, Element, ElementConversion, IndexingUpdateOp, Shape, SliceArg,
};

use crate::{
    Bool, BroadcastArgs, Device, Int, ReshapeArgs, Tensor, TensorCreationOptions, bool_and_impl,
    bool_or_impl,
    check::TensorCheck,
    kind::{Basic, Numeric},
    ops::{BasicOps, CompoundTensorKind, FloatMathOps}, split::base::SplitTensor,
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
impl<const D: usize, K> SplitTensor<D, K>
where
    K: Basic + CompoundTensorKind,
    K::Inner: Basic,
{
    /// Select complex tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices of the elements to select.
    pub fn select(mut self, dim: usize, indices: Tensor<1, Int>) -> Self {
        let indices = indices.primitive;
        crate::split_tensor_unary_body!(K, select, self, dim, indices)
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
        mut self,
        dim: usize,
        indices: Tensor<1, Int>,
        mut values: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        let indices = indices.primitive;

        let lhs_slice = self.components.as_mut();
        let rhs_slice = values.components.as_mut();
        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());
        if let Some(((lhs_last, lhs_head), (rhs_last, rhs_head))) =
            lhs_slice.split_last_mut().zip(rhs_slice.split_last_mut())
        {
            for (lhs_comp, rhs_comp) in lhs_head.iter_mut().zip(rhs_head.iter_mut()) {
                crate::overwrite_in_place!(lhs_comp, rhs_comp, |lhs, rhs| {
                    K::Inner::select_assign(lhs, dim, indices.clone(), rhs, update)
                });
            }
            crate::overwrite_in_place!(lhs_last, rhs_last, |lhs, rhs| {
                K::Inner::select_assign(lhs, dim, indices, rhs, update)
            });
        }
        self
    }

    /// Transpose the complex tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions.
    pub fn transpose(mut self) -> Self {
        crate::split_tensor_unary_body!(K, transpose, self)
    }

    /// Swaps two dimensions of a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    pub fn swap_dims(mut self, dim1: usize, dim2: usize) -> Self {
        crate::split_tensor_unary_body!(K, swap_dims, self, dim1, dim2)
    }

    /// Returns the device of the current complex tensor.
    pub fn device(&self) -> Device {
        K::device(&self.components.as_ref()[0])
    }

    /// Move the complex tensor to the given device.
    pub fn to_device(mut self, device: &Device) -> Self {
        crate::split_tensor_unary_body!(K, to_device, self, device)
    }

    /// Repeat the complex tensor along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension.
    pub fn repeat_dim(mut self, dim: usize, times: usize) -> Self {
        crate::split_tensor_unary_body!(K, repeat_dim, self, dim, times)
    }
    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are equal and `false` elsewhere.
    pub fn equal(mut self, mut rhs: Self) -> Tensor<D, Bool> {
        let lhs_slice = self.components.as_mut();
        let rhs_slice = rhs.components.as_mut();

        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());

        // 1. Isolate the first pair to initialize our accumulator tensor.
        let mut acc = if let Some((lhs_first, rhs_first)) =
            lhs_slice.first_mut().zip(rhs_slice.first_mut())
        {
            unsafe { K::Inner::equal(core::ptr::read(lhs_first), core::ptr::read(rhs_first)) }
        } else {
            // the compile time check on compound tensor len should prevent this from ever being possible
            unreachable!("Cannot check equality on a CompoundTensor with 0 components.");
        };

        // 2. Loop through the remaining pairs and fold them into the accumulator
        if lhs_slice.len() > 1 {
            for (lhs_comp, rhs_comp) in lhs_slice[1..].iter_mut().zip(rhs_slice[1..].iter_mut()) {
                acc = bool_and_impl(acc, unsafe {
                    K::Inner::equal(core::ptr::read(lhs_comp), core::ptr::read(rhs_comp))
                });
            }
        }
        Tensor::new(acc)
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are not equal and `false` elsewhere.
    pub fn not_equal(mut self, mut rhs: Self) -> Tensor<D, Bool> {
        let lhs_slice = self.components.as_mut();
        let rhs_slice = rhs.components.as_mut();

        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());

        // 1. Isolate the first pair to initialize our accumulator tensor.
        //    This avoids needing an "empty" or "dummy" starting boolean tensor.
        let mut acc = if let Some((lhs_first, rhs_first)) =
            lhs_slice.first_mut().zip(rhs_slice.first_mut())
        {
            unsafe { K::Inner::not_equal(core::ptr::read(lhs_first), core::ptr::read(rhs_first)) }
        } else {
            // the compile time check on compound tensor len should prevent this from ever being possible
            unreachable!("Cannot check equality on a CompoundTensor with 0 components.");
        };

        // 2. Loop through the remaining pairs and fold them into the accumulator
        if lhs_slice.len() > 1 {
            for (lhs_comp, rhs_comp) in lhs_slice[1..].iter_mut().zip(rhs_slice[1..].iter_mut()) {
                acc = bool_or_impl(acc, unsafe {
                    K::Inner::not_equal(core::ptr::read(lhs_comp), core::ptr::read(rhs_comp))
                });
            }
        }
        Tensor::new(acc)
    }

    /// Tests if any element in the complex tensor evaluates to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if any element is non-zero, `false` otherwise.
    pub fn any(mut self) -> Tensor<1, Bool> {
        let components = self.components.as_mut();

        let mut acc = if let Some(first) = components.first_mut() {
            unsafe { K::Inner::any(core::ptr::read(first)) }
        } else {
            // the compile time check on compound tensor len should prevent this from ever being possible
            unreachable!("Cannot check equality on a CompoundTensor with 0 components.");
        };

        if components.len() > 1 {
            for component in components[1..].iter_mut() {
                acc = bool_or_impl(acc, unsafe { K::Inner::any(core::ptr::read(component)) });
            }
        }

        Tensor::new(acc)
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
    pub fn any_dim(mut self, dim: usize) -> Tensor<D, Bool> {
        let components = self.components.as_mut();

        let mut acc = if let Some(first) = components.first_mut() {
            unsafe { K::Inner::any_dim(core::ptr::read(first), dim) }
        } else {
            // the compile time check on compound tensor len should prevent this from ever being possible
            unreachable!("Cannot check equality on a CompoundTensor with 0 components.");
        };

        if components.len() > 1 {
            for component in components[1..].iter_mut() {
                acc = bool_or_impl(acc, unsafe {
                    K::Inner::any_dim(core::ptr::read(component), dim)
                });
            }
        }

        Tensor::new(acc)
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if all elements are non-zero, `false` otherwise.
    pub fn all(mut self) -> Tensor<1, Bool> {
        let components = self.components.as_mut();

        let mut acc = if let Some(first) = components.first_mut() {
            unsafe { K::not_equal_elem(ptr::read(first), burn_std::Scalar::Float(0.0)) }
        } else {
            panic!("Cannot compute all on a CompoundTensor with 0 components.");
        };

        if components.len() > 1 {
            for component in components[1..].iter_mut() {
                acc = bool_and_impl(acc, unsafe {
                    K::not_equal_elem(ptr::read(component), burn_std::Scalar::Float(0.0))
                });
            }
        }

        Tensor::<1, Bool>::new(acc).all()
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
    pub fn all_dim(mut self, dim: usize) -> Tensor<D, Bool> {
        let components = self.components.as_mut();

        let mut acc = if let Some(first) = components.first_mut() {
            unsafe { K::not_equal_elem(ptr::read(first), burn_std::Scalar::Float(0.0)) }
        } else {
            // the compile time check on compound tensor len should prevent this from ever being possible
            unreachable!("Cannot compute all_dim on a CompoundTensor with 0 components.");
        };

        if components.len() > 1 {
            for component in components[1..].iter_mut() {
                acc = bool_and_impl(acc, unsafe {
                    K::not_equal_elem(ptr::read(component), burn_std::Scalar::Float(0.0))
                });
            }
        }

        Tensor::<D, Bool>::new(K::all_dim(acc, dim))
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
    pub fn permute<Dim>(mut self, axes: [Dim; D]) -> Self
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
            let device = self.device();
            let _dtype = inner_dtype::<K>(&device);
            let _shape = self.shape();
            crate::split_tensor_unary_body!(K, permute, self, &fixed_axes)
        }
    }

    /// Reverse the order of elements in the complex tensor along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The dimensions to reverse. The values must be unique and in the range of the
    ///   number of dimensions. Negative values are used as an offset from the end.
    pub fn flip<const N: usize>(mut self, axes: [isize; N]) -> Self {
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
        crate::split_tensor_unary_body!(K::Inner, flip, self, &transformed_axes)
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

        let values_shape = values.shape();
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

        // crate::split_tensor_binary_body!(
        //     kind: K,
        //     lhs: self,
        //     rhs: values,
        //     args:(slice),
        //     op: |comp_self, comp_values| K::slice_assign(comp_self, slice, comp_values)
        // )
        todo!()
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
    pub fn mask_where(mut self, mask: Tensor<D, Bool>, mut source: Self) -> Self {
        let mask_ref = mask.primitive;

        crate::split_tensor_binary_body!(
            kind: K::Inner,
            lhs: self,
            rhs: source,
            args:(mask_ref),
            op: |comp_self, comp_source| K::Inner::mask_where(comp_self, mask_ref, comp_source)
        )
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
    pub fn gather(mut self, dim: usize, indices: Tensor<D, Int>) -> Self {
        crate::check!(TensorCheck::gather::<D>(
            dim,
            &self.shape(),
            &indices.shape()
        ));
        let indices = indices.primitive;
        {
            let components_slice = self.components.as_mut();
            if let Some((last, head)) = components_slice.split_last_mut() {
                for component in head {
                    *component = K::Inner::gather(
                        dim,
                        unsafe { core::ptr::read(component) },
                        indices.clone(),
                    );
                }
                *last = K::Inner::gather(dim, unsafe { core::ptr::read(last) }, indices);
            }
            self
        }
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
        mut self,
        dim: usize,
        indices: Tensor<D, Int>,
        mut values: Self,
        update: burn_std::IndexingUpdateOp,
    ) -> Self {
        let indices = indices.primitive;
        let lhs_slice = self.components.as_mut();
        let rhs_slice = values.components.as_mut();
        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());
        if let Some(((lhs_last, lhs_head), (rhs_last, rhs_head))) =
            lhs_slice.split_last_mut().zip(rhs_slice.split_last_mut())
        {
            for (lhs_comp, rhs_comp) in lhs_head.iter_mut().zip(rhs_head.iter_mut()) {
                *lhs_comp = K::Inner::select_assign(
                    unsafe { core::ptr::read(lhs_comp) },
                    dim,
                    indices.clone(),
                    unsafe { core::ptr::read(rhs_comp) },
                    update,
                );
            }

            *lhs_last = K::Inner::select_assign(
                unsafe { core::ptr::read(lhs_last) },
                dim,
                indices.clone(),
                unsafe { core::ptr::read(rhs_last) },
                update,
            );
        }
        self
    }

    /// Create a complex tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The complex value to fill the tensor with.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn full<S: Into<Shape>, E: ElementConversion>(
        _shape: S,
        _fill_value: E,
        _options: impl Into<TensorCreationOptions>,
    ) -> Self {
        //TODO: figure out how to map dtype so that it doesn't just assume Complex<f64>
        // let e = E::elem::<Complex<f64>>(fill_value);
        // let shape = shape.into();
        // SplitTensor::from_parts_data(
        //     TensorData::full(&shape, e.real()),
        //     TensorData::full(&shape, e.imag()),
        //     &options.into().device,
        // )
        todo!()
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
        values: SplitTensor<DV, K>,
        _update: IndexingUpdateOp,
    ) -> Self {
        crate::check!(TensorCheck::scatter_nd::<D, M, DV>(
            &self.shape(),
            &indices.shape(),
            &values.shape()
        ));
        let _components = self.components;
        let _values_components = values.components;

        // for component in 0..components.len()-1 {
        //     components[component] = K::scatter_nd(
        //         components[component],
        //         indices.primitive.clone(),
        //         values_components[component],
        //         update.clone(),
        //     );
        // }

        // let last = components.len()-1;
        // components[last] = K::scatter_nd(
        //     components[last],
        //     indices.primitive,
        //     values_components[last],
        //     update,
        // );

        // Self {
        //     _kind: core::marker::PhantomData,
        //     components,
        // }
        todo!()
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
    ) -> SplitTensor<DV, K> {
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
impl<const D: usize, K: Numeric + BasicOps + CompoundTensorKind> SplitTensor<D, K>
where
    K::Inner: Numeric,
{
    /// Applies element-wise addition operation.
    ///
    /// `y = x2 + x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to add.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, mut rhs: Self) -> Self {
        crate::split_tensor_binary_body!(
            kind: K,
            lhs: self,
            rhs: rhs,
            args: (),
            op: |comp_self, comp_rhs| K::add(comp_self, comp_rhs)
        )
    }

    /// Applies element-wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to add, element-wise.
    pub fn add_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<ComplexScalar<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, &device);
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
    pub fn sub(mut self, mut rhs: Self) -> Self {
        crate::split_tensor_binary_body!(
            kind: K,
            lhs: self,
            rhs: rhs,
            args: (),
            op: |comp_self, comp_rhs| K::sub(comp_self, comp_rhs)
        )
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
    pub fn neg(mut self) -> Self {
        crate::split_tensor_unary_body!(K, neg, self)
    }

    /// Aggregate all elements in the complex tensor with the sum operation.
    pub fn sum(mut self) -> Self {
        crate::split_tensor_unary_body!(K, sum, self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn sum_dim(mut self, dim: usize) -> Self {
        crate::split_tensor_unary_body!(K, sum_dim, self, dim)
    }

    /// Aggregate all elements in the complex tensor with the mean operation.
    pub fn mean(mut self) -> Self {
        crate::split_tensor_unary_body!(K, mean, self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn mean_dim(mut self, dim: usize) -> Self {
        crate::split_tensor_unary_body!(K, mean_dim, self, dim)
    }

    /// Computes the cumulative sum of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    pub fn cumsum(mut self, dim: usize) -> Self {
        crate::split_tensor_unary_body!(K, cumsum, self, dim)
    }
}

#[macro_export]
macro_rules! split_tensor_unary_body {
    ($kind:ty, $func:ident, $self:expr $(, $arg_expr:expr)*) => {{
        let components_slice = $self.components.as_mut();

        if let Some((last, head)) = components_slice.split_last_mut() {
            // 1. Loop through all components except the last one
            for component in head {

                *component = <$kind>::$func(
                    unsafe { core::ptr::read(component) }
                    $(, (&$arg_expr).clone())*
                );
            }

            // 2. Handle the last component, consuming the original arguments

            *last = <$kind>::$func(
                unsafe { core::ptr::read(last) }
                $(, $arg_expr)*
            );
        }

        $self
    }};
}

#[macro_export]
macro_rules! split_tensor_binary_body {
    (
        kind: $kind:path,
        lhs: $lhs:expr,
        rhs: $rhs:expr,
        args: ($($arg:ident),*),
        op: |$l_placeholder:ident, $r_placeholder:ident| $op_expr:expr
    ) => {{
        let lhs_slice = $lhs.components.as_mut();
        let rhs_slice = $rhs.components.as_mut();

        debug_assert_eq!(lhs_slice.len(), rhs_slice.len());

        if let Some(((lhs_last, lhs_head), (rhs_last, rhs_head))) =
            lhs_slice.split_last_mut().zip(rhs_slice.split_last_mut())
        {
            // 1. Process all intermediate elements
            for (lhs_comp, rhs_comp) in lhs_head.iter_mut().zip(rhs_head.iter_mut()) {
                unsafe {
                    let $l_placeholder = core::ptr::read(lhs_comp);
                    let $r_placeholder = core::ptr::read(rhs_comp);

                    // Safely creates local clones of args ONLY if args are present
                    $( let $arg = $arg.clone(); )*

                    core::ptr::write(lhs_comp, $op_expr);
                }
            }

            // 2. Process the final elements, safely consuming the original values
            unsafe {
                let $l_placeholder = core::ptr::read(lhs_last);
                let $r_placeholder = core::ptr::read(rhs_last);

                core::ptr::write(lhs_last, $op_expr);
            }
        }

        $lhs
    }};
}

#[macro_export]
macro_rules! overwrite_in_place {
    ($lhs_slot:expr, $rhs_slot:expr, |$l_val:ident, $r_val:ident| $op:expr) => {
        unsafe {
            let $l_val = core::ptr::read($lhs_slot);
            let $r_val = core::ptr::read($rhs_slot);
            core::ptr::write($lhs_slot, $op);
        }
    };
}
