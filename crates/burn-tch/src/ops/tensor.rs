use super::TchOps;
use crate::{element::TchElement, LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor};
use burn_tensor::{
    backend::Backend,
    ops::{FloatTensorOps, IntTensor},
    Distribution, ElementConversion, Shape, TensorData,
};
use std::ops::Range;

impl<E: TchElement, Q: QuantElement> FloatTensorOps<Self> for LibTorch<E, Q> {
    fn float_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor<E> {
        TchTensor::from_data(data, (*device).into())
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &LibTorchDevice,
    ) -> TchTensor<E> {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::<E>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.rand_like_out(tensor))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<E>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<E>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::<E>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn float_repeat_dim(tensor: TchTensor<E>, dim: usize, times: usize) -> TchTensor<E> {
        TchOps::repeat_dim(tensor, dim, times)
    }

    fn float_zeros(shape: Shape, device: &LibTorchDevice) -> TchTensor<E> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (E::KIND, device)))
    }

    fn float_ones(shape: Shape, device: &LibTorchDevice) -> TchTensor<E> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (E::KIND, device)))
    }

    fn float_shape(tensor: &TchTensor<E>) -> Shape {
        tensor.shape()
    }

    async fn float_into_data(tensor: TchTensor<E>) -> TensorData {
        let shape = Self::float_shape(&tensor);
        let tensor = Self::float_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<E>, tch::TchError> = tensor.tensor.try_into();

        TensorData::new(values.unwrap(), shape)
    }

    fn float_device(tensor: &TchTensor<E>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn float_to_device(tensor: TchTensor<E>, device: &LibTorchDevice) -> TchTensor<E> {
        TchOps::to_device(tensor, device)
    }

    fn float_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor<E> {
        let tensor = tch::Tensor::empty(TchShape::from(shape).dims, (E::KIND, (*device).into()));

        TchTensor::new(tensor)
    }

    fn float_add(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchOps::add(lhs, rhs)
    }

    fn float_add_scalar(lhs: TchTensor<E>, rhs: E) -> TchTensor<E> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn float_sub(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchOps::sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: TchTensor<E>, rhs: E) -> TchTensor<E> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn float_mul(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchOps::mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: TchTensor<E>, rhs: E) -> TchTensor<E> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn float_div(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchOps::div(lhs, rhs)
    }

    fn float_div_scalar(lhs: TchTensor<E>, rhs: E) -> TchTensor<E> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        )
    }

    fn float_remainder_scalar(lhs: TchTensor<E>, rhs: E) -> TchTensor<E> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |tensor| tensor.f_remainder(rhs).unwrap(),
            |tensor| tensor.f_remainder(rhs).unwrap(),
        )
    }

    fn float_matmul(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        TchTensor::new(tensor)
    }

    fn float_neg(tensor: TchTensor<E>) -> TchTensor<E> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip(tensor: TchTensor<E>) -> TchTensor<E> {
        TchTensor::new(tensor.tensor.reciprocal())
    }

    fn float_swap_dims(tensor: TchTensor<E>, dim1: usize, dim2: usize) -> TchTensor<E> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape(tensor: TchTensor<E>, shape: Shape) -> TchTensor<E> {
        TchOps::reshape(tensor, shape)
    }

    fn float_gather(dim: usize, tensor: TchTensor<E>, indices: TchTensor<i64>) -> TchTensor<E> {
        TchOps::gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: TchTensor<E>,
        indices: TchTensor<i64>,
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn float_select(tensor: TchTensor<E>, dim: usize, indices: TchTensor<i64>) -> TchTensor<E> {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: TchTensor<E>,
        dim: usize,
        indices: TchTensor<i64>,
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn float_slice(tensor: TchTensor<E>, ranges: &[Range<usize>]) -> TchTensor<E> {
        TchOps::slice(tensor, ranges)
    }

    fn float_slice_assign(
        tensor: TchTensor<E>,
        ranges: &[Range<usize>],
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: TchTensor<E>,
        mask: TchTensor<bool>,
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        let output = value.tensor.where_self(&mask.tensor, &tensor.tensor);

        TchTensor::new(output)
    }

    fn float_mask_fill(tensor: TchTensor<E>, mask: TchTensor<bool>, value: E) -> TchTensor<E> {
        let value: f64 = value.elem();

        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn float_equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchOps::equal(lhs, rhs)
    }

    fn float_equal_elem(lhs: TchTensor<E>, rhs: E) -> TchTensor<bool> {
        TchOps::equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchOps::greater(lhs, rhs)
    }

    fn float_greater_elem(lhs: TchTensor<E>, rhs: E) -> TchTensor<bool> {
        TchOps::greater_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater_equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(lhs: TchTensor<E>, rhs: E) -> TchTensor<bool> {
        TchOps::greater_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchOps::lower(lhs, rhs)
    }

    fn float_lower_elem(lhs: TchTensor<E>, rhs: E) -> TchTensor<bool> {
        TchOps::lower_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower_equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(lhs: TchTensor<E>, rhs: E) -> TchTensor<bool> {
        TchOps::lower_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_mean(tensor: TchTensor<E>) -> TchTensor<E> {
        TchOps::mean(tensor)
    }

    fn float_sum(tensor: TchTensor<E>) -> TchTensor<E> {
        TchOps::sum(tensor)
    }

    fn float_sum_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchOps::sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchOps::mean_dim(tensor, dim)
    }

    fn float_prod(tensor: TchTensor<E>) -> TchTensor<E> {
        TchOps::prod(tensor)
    }

    fn float_prod_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchOps::prod_dim(tensor, dim)
    }

    fn float_argmax(tensor: TchTensor<E>, dim: usize) -> TchTensor<i64> {
        TchOps::argmax(tensor, dim)
    }

    fn float_argmin(tensor: TchTensor<E>, dim: usize) -> TchTensor<i64> {
        TchOps::argmin(tensor, dim)
    }

    fn float_max_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchOps::max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn float_min_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchOps::min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn float_exp(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.exp_(), |tensor| tensor.exp())
    }

    fn float_log(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.log_(), |tensor| tensor.log())
    }

    fn float_log1p(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.log1p_(), |tensor| tensor.log1p())
    }

    fn float_powf_scalar(tensor: TchTensor<E>, value: f32) -> TchTensor<E> {
        tensor.unary_ops(
            |mut tensor| tensor.f_pow_(value as f64).unwrap(),
            |tensor| tensor.pow_tensor_scalar(value as f64),
        )
    }

    fn float_sqrt(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.sqrt_(), |tensor| tensor.sqrt())
    }

    fn float_abs(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn float_cos(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.cos_(), |tensor| tensor.cos())
    }

    fn float_sin(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.sin_(), |tensor| tensor.sin())
    }

    fn float_tanh(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.tanh_(), |tensor| tensor.tanh())
    }

    fn float_erf(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.erf_(), |tensor| tensor.erf())
    }

    fn float_cat(tensors: Vec<TchTensor<E>>, dim: usize) -> TchTensor<E> {
        TchOps::cat(tensors, dim)
    }

    fn float_clamp_min(tensor: TchTensor<E>, min: E) -> TchTensor<E> {
        TchOps::clamp_min(tensor, min.elem::<f64>())
    }

    fn float_clamp_max(
        tensor: TchTensor<E>,
        max: <LibTorch<E> as Backend>::FloatElem,
    ) -> TchTensor<E> {
        TchOps::clamp_max(tensor, max.elem::<f64>())
    }

    fn float_clamp(
        tensor: TchTensor<E>,
        min: <LibTorch<E> as Backend>::FloatElem,
        max: <LibTorch<E> as Backend>::FloatElem,
    ) -> TchTensor<E> {
        TchOps::clamp(tensor, min.elem::<f64>(), max.elem::<f64>())
    }

    fn float_into_int(tensor: TchTensor<E>) -> TchTensor<i64> {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn float_narrow(tensor: TchTensor<E>, dim: usize, start: usize, length: usize) -> TchTensor<E> {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn float_chunk(tensor: TchTensor<E>, chunks: usize, dim: usize) -> Vec<TchTensor<E>> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn float_powf(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchOps::powf(lhs, rhs)
    }

    fn float_permute(tensor: TchTensor<E>, axes: &[usize]) -> TchTensor<E> {
        TchOps::permute(tensor, axes)
    }

    fn float_flip(tensor: TchTensor<E>, axes: &[usize]) -> TchTensor<E> {
        TchOps::flip(tensor, axes)
    }

    fn float_sign(tensor: TchTensor<E>) -> TchTensor<E> {
        TchOps::sign(tensor)
    }

    fn float_expand(tensor: TchTensor<E>, shape: Shape) -> TchTensor<E> {
        TchOps::expand(tensor, shape)
    }

    fn float_sort(tensor: TchTensor<E>, dim: usize, descending: bool) -> TchTensor<E> {
        TchOps::sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
        descending: bool,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        TchOps::sort_with_indices(tensor, dim, descending)
    }

    fn float_argsort(tensor: TchTensor<E>, dim: usize, descending: bool) -> IntTensor<Self> {
        TchOps::argsort(tensor, dim, descending)
    }
}
