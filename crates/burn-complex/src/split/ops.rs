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
    pub fn empty<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions<B>>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        let dtype = opt.resolve_dtype::<Float>();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        SplitBackend::<B, D>::complex_zeros(shape, &opt.device, dtype.into())
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
        SplitBackend::<B, D>::complex_slice(self, &slices)
    }

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

    pub fn transpose(self) -> Self {
        SplitBackend::<B, D>::complex_transpose(self)
    }

    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitBackend::<B, D>::complex_swap_dims(self, dim1, dim2)
    }

    pub fn device(&self) -> B::Device {
        SplitBackend::<B, D>::complex_device(self)
    }

    pub fn to_device(self, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_to_device(self, device)
    }

    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        SplitBackend::<B, D>::complex_into_interleaved_data(self).await
    }

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

    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitBackend::<B, D>::complex_repeat_dim(self, dim, times)
    }
    pub fn equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_equal(self, rhs, out_dtype)
    }

    pub fn not_equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_not_equal(self, rhs, out_dtype)
    }

    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cat(tensors, dim)
    }

    pub fn any(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_any(self, out_dtype)
    }

    pub fn any_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_any_dim(self, dim, out_dtype)
    }

    pub fn all(self) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_all(self, out_dtype)
    }

    pub fn all_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        SplitBackend::<B, D>::complex_all_dim(self, dim, out_dtype)
    }

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

    pub fn mask_where(self, mask: Tensor<B, D, Bool>, source: Self) -> Self {
        SplitBackend::<B, D>::complex_mask_where(self, mask.into_primitive(), source)
    }

    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<B, D, Bool>, value: E) -> Self {
        SplitBackend::<B, D>::complex_mask_fill(self, mask.into_primitive(), value.elem())
    }

    pub fn gather(self, dim: usize, indices: Tensor<B, D, Int>) -> Self {
        // check!(TensorCheck::gather::<D>(
        //     dim,
        //     &self.shape(),
        //     &indices.shape()
        // ));
        SplitBackend::<B, D>::complex_gather(dim, self, indices.into_primitive())
    }

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

    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
    }

    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<B, D, Bool> {
        let out_dtype =
            get_device_settings::<B>(&SplitBackend::<B, D>::complex_device(&self)).bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_not_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
    }

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
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_add(self, rhs)
    }

    pub fn add_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_add(self, scalar_tensor)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_sub(self, rhs)
    }

    pub fn sub_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_sub(self, scalar_tensor)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_div(self, rhs)
    }

    pub fn div_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_div(self, scalar_tensor)
    }

    pub fn remainder(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_remainder(self, rhs)
    }

    pub fn remainder_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        SplitBackend::<B, D>::complex_remainder_scalar(self, rhs.elem())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_mul(self, rhs)
    }

    pub fn mul_scalar(self, rhs: burn_tensor::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_mul(self, scalar_tensor)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        SplitBackend::<B, D>::complex_neg(self)
    }

    pub fn sign(self) -> Self {
        SplitBackend::<B, D>::complex_sign(self)
    }

    pub fn sum(self) -> Self {
        SplitBackend::<B, D>::complex_sum(self)
    }

    pub fn sum_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_sum_dim(self, dim)
    }

    pub fn prod(self) -> Self {
        SplitBackend::<B, D>::complex_prod(self)
    }

    pub fn prod_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_prod_dim(self, dim)
    }

    pub fn mean(self) -> Self {
        SplitBackend::<B, D>::complex_mean(self)
    }

    pub fn mean_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_mean_dim(self, dim)
    }

    pub fn cumsum(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumsum(self, dim)
    }

    pub fn cumprod(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumprod(self, dim)
    }

    pub fn powi(self, other: Tensor<B, D, Int>) -> Self {
        SplitBackend::<B, D>::complex_powi(self, other.into_primitive())
    }

    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(other, &self.dtype());
        SplitBackend::<B, D>::complex_powi_scalar(self, other)
    }

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
    pub fn conj(self) -> Self {
        SplitBackend::<B, D>::conj(self)
    }

    pub fn phase(self) -> F {
        SplitBackend::<B, D>::complex_arg(self)
    }

    pub fn magnitude(self) -> F {
        SplitBackend::<B, D>::abs(self)
    }

    pub fn exp(self) -> Self {
        SplitBackend::<B, D>::complex_exp(self)
    }

    pub fn sin(self) -> Self {
        SplitBackend::<B, D>::complex_sin(self)
    }

    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T) -> Self {
        SplitBackend::<B, D>::complex_from_parts(real.into(), imag.into())
    }

    pub fn from_interleaved_data(data: TensorData, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_from_interleaved_data(data, device)
    }

    pub fn from_polar(magnitude: F, phase: F) -> Self {
        SplitBackend::<B, D>::complex_from_polar(magnitude, phase)
    }

    pub fn cos(self) -> Self {
        SplitBackend::<B, D>::complex_cos(self)
    }

    pub fn log(self) -> Self {
        SplitBackend::<B, D>::complex_log(self)
    }
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
