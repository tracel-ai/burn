use alloc::vec::Vec;
use burn_std::{DType, Shape, SliceArg};
use burn_tensor::{
    Device, Float, IndexingUpdateOp, TensorCreationOptions, TensorData, TensorMetadata,
    backend::{Backend, BackendTypes, ExecutionError},
    get_device_settings, try_read_sync,
};

use crate::{
    base::{ComplexTensorBackend, ComplexTensorOps},
    split::{SplitBackend, SplitComplexTensor},
};

impl<B, F> SplitComplexTensor<B, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    pub fn empty<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        SplitBackend::<B>::complex_zeros(shape, &opt.device, dtype.into())
    }

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
        SplitBackend::<B>::complex_slice(self, &slices)
    }

    pub fn zeros<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let options = options.into();
        let shape = shape.into();
        let device = &options.device;
        let dtype = crate::utils::real_to_complex_dtype(options.resolve_dtype::<Float>());
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B>::complex_zeros(shape, device, dtype.into())
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }
}
//BasicOps
impl<B, F> SplitComplexTensor<B, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    pub fn select(self, dim: usize, indices: B::IntTensorPrimitive) -> Self {
        // Uses your existing `select` name.
        SplitBackend::<B>::complex_select(self, dim, indices)
    }

    pub fn select_assign(
        self,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => {
                SplitBackend::<B>::complex_select_add(self, dim, indices, values)
            }
            _ => unimplemented!(),
        }
    }

    pub fn reshape(self, shape: Shape) -> Self {
        SplitBackend::<B>::complex_reshape(self, shape)
    }

    pub fn transpose(self) -> Self {
        SplitBackend::<B>::complex_transpose(self)
    }

    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitBackend::<B>::complex_swap_dims(self, dim1, dim2)
    }

    pub fn device(&self) -> B::Device {
        SplitBackend::<B>::complex_device(self)
    }

    pub fn to_device(self, device: &B::Device) -> Self {
        SplitBackend::<B>::complex_to_device(self, device)
    }

    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        SplitBackend::<B>::complex_into_interleaved_data(self).await
    }

    pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions<B>>) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        let opt = options.into();
        SplitBackend::<B>::complex_from_interleaved_data(
            data.convert::<<SplitBackend<B> as ComplexTensorBackend>::ComplexScalar>(),
            &opt.device,
        )
    }

    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitBackend::<B>::complex_repeat_dim(self, dim, times)
    }
    pub fn equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_equal(self, rhs, out_dtype)
    }

    pub fn not_equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_not_equal(self, rhs, out_dtype)
    }

    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        SplitBackend::<B>::complex_cat(tensors, dim)
    }

    pub fn any(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_any(self, out_dtype)
    }

    pub fn any_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_any_dim(self, dim, out_dtype)
    }

    pub fn all(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_all(self, out_dtype)
    }

    pub fn all_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_all_dim(self, dim, out_dtype)
    }

    pub fn permute<Dim>(self, axes: [Dim; D]) -> Self
    where
        Dim: AsIndex,
    {
        SplitBackend::<B>::complex_permute(self, axes)
    }

    pub fn expand(self, shape: Shape) -> Self {
        SplitBackend::<B>::complex_expand(self, shape)
    }

    pub fn flip(self, axes: &[usize]) -> Self {
        SplitBackend::<B>::complex_flip(self, axes)
    }

    pub fn unfold(self, dim: usize, size: usize, step: usize) -> Self {
        SplitBackend::<B>::complex_unfold(self, dim, size, step)
    }

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

        let values_shape = SplitBackend::<B>::complex_shape(&values);
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

        SplitBackend::<B>::complex_slice_assign(self, &slices, values)
    }

    pub fn ones(shape: Shape, device: &B::Device, dtype: DType) -> Self {
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B>::complex_ones(shape, device, dtype.into())
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }

    pub fn mask_where(self, mask: B::BoolTensorPrimitive, source: Self) -> Self {
        SplitBackend::<B>::complex_mask_where(self, mask, source)
    }

    pub fn mask_fill(self, mask: B::BoolTensorPrimitive, value: burn_tensor::Scalar) -> Self {
        SplitBackend::<B>::complex_mask_fill(self, mask, value.elem())
    }

    pub fn gather(self, dim: usize, indices: B::IntTensorPrimitive) -> Self {
        SplitBackend::<B>::complex_gather(dim, self, indices)
    }

    pub fn scatter(
        self,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: Self,
        update: burn_tensor::IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => {
                SplitBackend::<B>::complex_scatter_add(dim, self, indices, values)
            }
            _ => unimplemented!(),
        }
    }

    pub fn equal_elem(self, rhs: burn_tensor::Scalar) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B>::complex_device(&self)).bool_dtype;
        SplitBackend::<B>::complex_equal_elem(self, rhs.elem(), out_dtype)
    }

    pub fn not_equal_elem(self, rhs: burn_tensor::Scalar) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<SplitBackend<B>>(&SplitBackend::<B>::complex_device(&self))
                .bool_dtype;
        SplitBackend::<B>::complex_not_equal_elem(self, rhs.elem(), out_dtype)
    }

    pub fn full(
        shape: Shape,
        fill_value: burn_tensor::Scalar,
        device: &Device<B>,
        dtype: DType,
    ) -> Self {
        // Enforce complex dtype for clarity (mirrors from_data_dtype below).
        if !dtype.is_complex() {
            panic!("Expected complex dtype, got {dtype:?}");
        }
        // `elem()` should yield something convertible to `B::ComplexElem`.
        SplitBackend::<B>::complex_full(shape, fill_value.elem(), device)
    }

    pub fn scatter_nd(
        self,
        indices: B::IntTensorPrimitive,
        values: Self,
        reduction: IndexingUpdateOp,
    ) -> Self {
        SplitBackend::<B>::complex_scatter_nd(self, indices, values, reduction)
    }

    pub fn gather_nd(self, indices: B::IntTensorPrimitive) -> Self {
        todo!()
    }
}
//impl<B, F> Numeric<B> for SplitComplexTensor<F>
impl<B: Backend, F> SplitComplexTensor<B, F>
where
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    pub fn add(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_add(self, rhs)
    }

    pub fn add_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B>::complex_device(&self);
        let shape = SplitBackend::<B>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B>::complex_add(self, scalar_tensor)
    }

    pub fn sub(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_sub(self, rhs)
    }

    pub fn sub_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B>::complex_device(&self);
        let shape = SplitBackend::<B>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B>::complex_sub(self, scalar_tensor)
    }

    pub fn div(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_div(self, rhs)
    }

    pub fn div_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B>::complex_device(&self);
        let shape = SplitBackend::<B>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B>::complex_div(self, scalar_tensor)
    }

    pub fn remainder(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_remainder(self, rhs)
    }

    pub fn remainder_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        SplitBackend::<B>::complex_remainder_scalar(self, rhs.elem())
    }

    pub fn mul(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_mul(self, rhs)
    }

    pub fn mul_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B>::complex_device(&self);
        let shape = SplitBackend::<B>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B>::complex_mul(self, scalar_tensor)
    }

    pub fn neg(self) -> Self {
        SplitBackend::<B>::complex_neg(self)
    }

    pub fn sign(self) -> Self {
        SplitBackend::<B>::complex_sign(self)
    }

    pub fn sum(self) -> Self {
        SplitBackend::<B>::complex_sum(self)
    }

    pub fn sum_dim(self, dim: usize) -> Self {
        SplitBackend::<B>::complex_sum_dim(self, dim)
    }

    pub fn prod(self) -> Self {
        SplitBackend::<B>::complex_prod(self)
    }

    pub fn prod_dim(self, dim: usize) -> Self {
        SplitBackend::<B>::complex_prod_dim(self, dim)
    }

    pub fn mean(self) -> Self {
        SplitBackend::<B>::complex_mean(self)
    }

    pub fn mean_dim(self, dim: usize) -> Self {
        SplitBackend::<B>::complex_mean_dim(self, dim)
    }

    pub fn cumsum(self, dim: usize) -> Self {
        SplitBackend::<B>::complex_cumsum(self, dim)
    }

    pub fn cumprod(self, dim: usize) -> Self {
        SplitBackend::<B>::complex_cumprod(self, dim)
    }

    pub fn powi(self, rhs: B::IntTensorPrimitive) -> Self {
        SplitBackend::<B>::complex_powi(self, rhs)
    }

    pub fn powi_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        SplitBackend::<B>::complex_powi_scalar(self, rhs)
    }

    pub fn random(
        shape: burn_std::Shape,
        distribution: burn_tensor::Distribution,
        device: &B::Device,
        dtype: burn_std::DType,
    ) -> Self {
        SplitBackend::<B>::complex_random(
            shape,
            distribution,
            device,
            burn_std::FloatDType::from(crate::utils::complex_to_real_dtype(dtype)),
        )
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

    pub fn matmul(self, rhs: Self) -> Self {
        SplitBackend::<B>::complex_matmul(self, rhs)
    }
}

// ComplexOnlyOps
impl<B, F> SplitComplexTensor<B, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    pub fn conj(self) -> Self {
        SplitBackend::<B>::conj(self)
    }

    pub fn phase(self) -> F {
        SplitBackend::<B>::complex_arg(self)
    }

    pub fn magnitude(self) -> F {
        SplitBackend::<B>::abs(self)
    }

    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T) -> Self {
        SplitBackend::<B>::complex_from_parts(real.into(), imag.into())
    }

    pub fn from_interleaved_data(data: TensorData, device: &B::Device) -> Self {
        SplitBackend::<B>::complex_from_interleaved_data(data, device)
    }

    pub fn from_polar(magnitude: F, phase: F) -> Self {
        SplitBackend::<B>::complex_from_polar(magnitude, phase)
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
