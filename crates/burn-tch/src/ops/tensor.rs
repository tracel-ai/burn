use super::TchOps;
use crate::{IntoKind, LibTorch, LibTorchDevice, TchShape, TchTensor, element::TchElement};
use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::{BoolTensor, FloatTensor};
use burn_tensor::{
    DType, Distribution, ElementConversion, FloatDType, Shape, TensorData, TensorMetadata,
    backend::Backend,
    ops::{FloatTensorOps, IntTensor},
};
use half::{bf16, f16};

impl<E: TchElement> FloatTensorOps<Self> for LibTorch<E> {
    fn float_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor {
        match data.dtype {
            DType::F64 => TchTensor::from_data::<f64>(data, (*device).into()),
            DType::F32 => TchTensor::from_data::<f32>(data, (*device).into()),
            DType::F16 => TchTensor::from_data::<f16>(data, (*device).into()),
            DType::BF16 => TchTensor::from_data::<bf16>(data, (*device).into()),
            _ => unimplemented!("Unsupported dtype for `float_from_data`"),
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &LibTorchDevice,
    ) -> TchTensor {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::empty::<E>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.rand_like_out(tensor))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::empty::<E>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::empty::<E>(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::empty::<E>(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn float_repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        TchOps::repeat_dim(tensor, dim, times)
    }

    fn float_zeros(shape: Shape, device: &LibTorchDevice, dtype: FloatDType) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (dtype.into_kind(), device)))
    }

    fn float_ones(shape: Shape, device: &LibTorchDevice, dtype: FloatDType) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (dtype.into_kind(), device)))
    }

    async fn float_into_data(tensor: TchTensor) -> Result<TensorData, ExecutionError> {
        let shape = tensor.shape();
        let tensor = Self::float_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        Ok(match tensor.tensor.kind() {
            tch::Kind::Half => {
                let values = Vec::<f16>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            tch::Kind::Float => {
                let values = Vec::<f32>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            tch::Kind::Double => {
                let values = Vec::<f64>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            tch::Kind::BFloat16 => {
                let values = Vec::<bf16>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            _ => panic!("Not a valid float kind"),
        })
    }

    fn float_device(tensor: &TchTensor) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn float_to_device(tensor: TchTensor, device: &LibTorchDevice) -> TchTensor {
        TchOps::to_device(tensor, device)
    }

    fn float_empty(
        shape: Shape,
        device: &<LibTorch<E> as Backend>::Device,
        dtype: FloatDType,
    ) -> TchTensor {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (dtype.into_kind(), (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn float_add(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::add(lhs, rhs)
    }

    fn float_add_scalar(lhs: TchTensor, rhs: E) -> TchTensor {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn float_sub(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: TchTensor, rhs: E) -> TchTensor {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn float_mul(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: TchTensor, rhs: E) -> TchTensor {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn float_div(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::div(lhs, rhs)
    }

    fn float_div_scalar(lhs: TchTensor, rhs: E) -> TchTensor {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        )
    }

    fn float_remainder(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::remainder(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: TchTensor, rhs: E) -> TchTensor {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |tensor| tensor.f_remainder(rhs).unwrap(),
            |tensor| tensor.f_remainder(rhs).unwrap(),
        )
    }

    fn float_matmul(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        TchTensor::new(tensor)
    }

    fn float_cross(lhs: TchTensor, rhs: TchTensor, dim: usize) -> TchTensor {
        let tensor = lhs.tensor.cross(&rhs.tensor, dim as i64);
        TchTensor::new(tensor)
    }

    fn float_neg(tensor: TchTensor) -> TchTensor {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip(tensor: TchTensor) -> TchTensor {
        TchTensor::new(tensor.tensor.reciprocal())
    }

    fn float_swap_dims(tensor: TchTensor, dim1: usize, dim2: usize) -> TchTensor {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::reshape(tensor, shape)
    }

    fn float_gather(dim: usize, tensor: TchTensor, indices: TchTensor) -> TchTensor {
        TchOps::gather(dim, tensor, indices)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: TchTensor,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn float_select(tensor: TchTensor, dim: usize, indices: TchTensor) -> TchTensor {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn float_select_add(
        tensor: TchTensor,
        dim: usize,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn float_slice(tensor: TchTensor, slices: &[burn_tensor::Slice]) -> TchTensor {
        TchOps::slice_with_steps(tensor, slices)
    }

    fn float_slice_assign(
        tensor: TchTensor,
        slices: &[burn_tensor::Slice],
        value: TchTensor,
    ) -> TchTensor {
        TchOps::slice_assign(tensor, slices, value)
    }

    fn float_mask_where(tensor: TchTensor, mask: TchTensor, value: TchTensor) -> TchTensor {
        let output = value.tensor.where_self(&mask.tensor, &tensor.tensor);

        TchTensor::new(output)
    }

    fn float_mask_fill(tensor: TchTensor, mask: TchTensor, value: E) -> TchTensor {
        let value: f64 = value.elem();

        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn float_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::equal(lhs, rhs)
    }

    fn float_equal_elem(lhs: TchTensor, rhs: E) -> TchTensor {
        TchOps::equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater(lhs, rhs)
    }

    fn float_greater_elem(lhs: TchTensor, rhs: E) -> TchTensor {
        TchOps::greater_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(lhs: TchTensor, rhs: E) -> TchTensor {
        TchOps::greater_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower(lhs, rhs)
    }

    fn float_lower_elem(lhs: TchTensor, rhs: E) -> TchTensor {
        TchOps::lower_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(lhs: TchTensor, rhs: E) -> TchTensor {
        TchOps::lower_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_mean(tensor: TchTensor) -> TchTensor {
        TchOps::mean(tensor)
    }

    fn float_sum(tensor: TchTensor) -> TchTensor {
        TchOps::sum(tensor)
    }

    fn float_sum_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::mean_dim(tensor, dim)
    }

    fn float_cumsum(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::cumsum(tensor, dim)
    }

    fn float_cumprod(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::cumprod(tensor, dim)
    }

    fn float_cummin(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::cummin(tensor, dim)
    }

    fn float_cummax(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::cummax(tensor, dim)
    }

    fn float_prod(tensor: TchTensor) -> TchTensor {
        TchOps::prod(tensor)
    }

    fn float_prod_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::prod_dim(tensor, dim)
    }

    fn float_argmax(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmax(tensor, dim)
    }

    fn float_argmin(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmin(tensor, dim)
    }

    fn float_max_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn float_min_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn float_exp(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.exp_(), |tensor| tensor.exp())
    }

    fn float_log(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.log_(), |tensor| tensor.log())
    }

    fn float_log1p(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.log1p_(), |tensor| tensor.log1p())
    }

    fn float_powf_scalar_impl(tensor: TchTensor, value: f32) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_pow_(value as f64).unwrap(),
            |tensor| tensor.pow_tensor_scalar(value as f64),
        )
    }

    fn float_sqrt(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.sqrt_(), |tensor| tensor.sqrt())
    }

    fn float_abs(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn float_cos(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.cos_(), |tensor| tensor.cos())
    }

    fn float_sin(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.sin_(), |tensor| tensor.sin())
    }

    fn float_tanh(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.tanh_(), |tensor| tensor.tanh())
    }

    fn float_round(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.round_(), |tensor| tensor.round())
    }

    fn float_floor(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.floor_(), |tensor| tensor.floor())
    }

    fn float_ceil(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.ceil_(), |tensor| tensor.ceil())
    }

    fn float_trunc(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.trunc_(), |tensor| tensor.trunc())
    }

    fn float_erf(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.erf_(), |tensor| tensor.erf())
    }

    fn float_cat(tensors: Vec<TchTensor>, dim: usize) -> TchTensor {
        TchOps::cat(tensors, dim)
    }

    fn float_clamp_min(tensor: TchTensor, min: E) -> TchTensor {
        TchOps::clamp_min(tensor, min.elem::<f64>())
    }

    fn float_clamp_max(tensor: TchTensor, max: <LibTorch<E> as Backend>::FloatElem) -> TchTensor {
        TchOps::clamp_max(tensor, max.elem::<f64>())
    }

    fn float_clamp(
        tensor: TchTensor,
        min: <LibTorch<E> as Backend>::FloatElem,
        max: <LibTorch<E> as Backend>::FloatElem,
    ) -> TchTensor {
        TchOps::clamp(tensor, min.elem::<f64>(), max.elem::<f64>())
    }

    fn float_into_int(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn float_powf(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::powf(lhs, rhs)
    }

    fn float_permute(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        TchOps::permute(tensor, axes)
    }

    fn float_flip(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        TchOps::flip(tensor, axes)
    }

    fn float_sign(tensor: TchTensor) -> TchTensor {
        TchOps::sign(tensor)
    }

    fn float_expand(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::expand(tensor, shape)
    }

    fn float_sort(tensor: TchTensor, dim: usize, descending: bool) -> TchTensor {
        TchOps::sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: TchTensor,
        dim: usize,
        descending: bool,
    ) -> (TchTensor, TchTensor) {
        TchOps::sort_with_indices(tensor, dim, descending)
    }

    fn float_argsort(tensor: TchTensor, dim: usize, descending: bool) -> IntTensor<Self> {
        TchOps::argsort(tensor, dim, descending)
    }

    fn float_cast(tensor: TchTensor, dtype: FloatDType) -> TchTensor {
        // NOTE: when dtypes of inputs to an arithmetic operation differ, tch handles type
        // promotion based on a set of rules: https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc

        // Type promotion is not automatic on all backends so this behavior might differ
        let kind = dtype.into_kind();

        if tensor.tensor.kind() == kind {
            tensor
        } else {
            TchTensor::new(tensor.tensor.to_kind(kind))
        }
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        TchOps::unfold(tensor, dim, size, step)
    }

    fn float_is_nan(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        TchTensor::new(tensor.tensor.isnan())
    }

    fn float_is_inf(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        TchTensor::new(tensor.tensor.isinf())
    }
}
