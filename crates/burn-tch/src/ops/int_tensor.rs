use std::ops::Range;

use burn_tensor::{
    backend::Backend,
    ops::{IntTensor, IntTensorOps},
    Distribution, Shape, TensorData, TensorMetadata,
};

use crate::{element::TchElement, LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement, Q: QuantElement> IntTensorOps<Self> for LibTorch<E, Q> {
    fn int_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor {
        TchTensor::from_data::<i64>(data, (*device).into())
    }

    fn int_repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn int_into_data(tensor: TchTensor) -> TensorData {
        let shape = tensor.shape();
        let tensor = Self::int_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<i64>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        TensorData::new(values.unwrap(), shape)
    }

    fn int_to_device(tensor: TchTensor, device: &LibTorchDevice) -> TchTensor {
        TchOps::to_device(tensor, device)
    }

    fn int_reshape(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::reshape(tensor, shape)
    }

    fn int_device(tensor: &TchTensor) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn int_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (tch::Kind::Int64, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn int_slice(tensor: TchTensor, ranges: &[Range<usize>]) -> TchTensor {
        TchOps::slice(tensor, ranges)
    }

    fn int_slice_assign(tensor: TchTensor, ranges: &[Range<usize>], value: TchTensor) -> TchTensor {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<TchTensor>, dim: usize) -> TchTensor {
        TchOps::cat(tensors, dim)
    }

    fn int_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: TchTensor, rhs: i64) -> TchTensor {
        TchOps::equal_elem(lhs, rhs)
    }

    fn int_greater(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: TchTensor, rhs: i64) -> TchTensor {
        TchOps::greater_elem(lhs, rhs)
    }

    fn int_greater_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: TchTensor, rhs: i64) -> TchTensor {
        TchOps::greater_equal_elem(lhs, rhs)
    }

    fn int_lower(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: TchTensor, rhs: i64) -> TchTensor {
        TchOps::lower_elem(lhs, rhs)
    }

    fn int_lower_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: TchTensor, rhs: i64) -> TchTensor {
        TchOps::lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::add(lhs, rhs)
    }

    fn int_add_scalar(lhs: TchTensor, rhs: i64) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn int_sub(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: TchTensor, rhs: i64) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn int_mul(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: TchTensor, rhs: i64) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn int_div(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));
        let rhs: TchTensor =
            TchTensor::new(rhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out = TchOps::div(lhs, rhs);

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn int_div_scalar(lhs: TchTensor, rhs: i64) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out: TchTensor = lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        );

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn int_remainder(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));
        let rhs: TchTensor =
            TchTensor::new(rhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out = TchOps::remainder(lhs, rhs);

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn int_remainder_scalar(lhs: TchTensor, rhs: i64) -> TchTensor {
        lhs.unary_ops(
            |tensor| tensor.f_remainder(rhs).unwrap(),
            |tensor| tensor.f_remainder(rhs).unwrap(),
        )
    }

    fn int_neg(tensor: TchTensor) -> TchTensor {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_ones(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_full(
        shape: Shape,
        fill_value: i64,
        device: &<LibTorch<E> as Backend>::Device,
    ) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::full(
            shape.dims,
            fill_value,
            (tch::Kind::Int64, device),
        ))
    }

    fn int_sum(tensor: TchTensor) -> TchTensor {
        TchOps::sum(tensor)
    }

    fn int_sum_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: TchTensor) -> TchTensor {
        TchOps::prod(tensor)
    }

    fn int_prod_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::prod_dim(tensor, dim)
    }

    fn int_mean(tensor: TchTensor) -> TchTensor {
        let tensor: TchTensor =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));
        let output: TchTensor = TchTensor::new(TchOps::mean(tensor).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn int_mean_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        let tensor: TchTensor =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));

        let output: TchTensor = TchTensor::new(TchOps::mean_dim(tensor, dim).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn int_gather(dim: usize, tensor: TchTensor, indices: TchTensor) -> TchTensor {
        TchOps::gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: TchTensor,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn int_select(tensor: TchTensor, dim: usize, indices: TchTensor) -> TchTensor {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: TchTensor,
        dim: usize,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn int_mask_where(tensor: TchTensor, mask: TchTensor, source: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            tensor,
            source,
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
        )
    }

    fn int_mask_fill(tensor: TchTensor, mask: TchTensor, value: i64) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn int_argmax(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmax(tensor, dim)
    }

    fn int_argmin(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmin(tensor, dim)
    }

    fn int_max_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn int_clamp_min(tensor: TchTensor, min: i64) -> TchTensor {
        TchOps::clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: TchTensor, max: i64) -> TchTensor {
        TchOps::clamp_max(tensor, max)
    }

    fn int_clamp(tensor: TchTensor, min: i64, max: i64) -> TchTensor {
        TchOps::clamp(tensor, min, max)
    }

    fn int_abs(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn int_into_float(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow(tensor: TchTensor, dim: usize, start: usize, length: usize) -> TchTensor {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn int_chunk(tensor: TchTensor, chunks: usize, dim: usize) -> Vec<TchTensor> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn int_split(tensor: TchTensor, split_size: usize, dim: usize) -> Vec<TchTensor> {
        TchOps::split(tensor, split_size, dim)
    }

    fn int_split_with_sizes(
        tensor: TchTensor,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<TchTensor> {
        TchOps::split_with_sizes(tensor, split_sizes, dim)
    }

    fn int_random(shape: Shape, distribution: Distribution, device: &LibTorchDevice) -> TchTensor {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.uniform_(0.0, 255.0))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn int_arange(range: Range<i64>, device: &LibTorchDevice) -> TchTensor {
        let device: tch::Device = (*device).into();
        let mut tensor = tch::Tensor::arange(range.end - range.start, (tch::Kind::Int64, device));

        if range.start != 0 {
            tensor = tensor.f_add_scalar_(range.start).unwrap();
        }

        TchTensor::new(tensor)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        TchOps::permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        TchOps::flip(tensor, axes)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::sign(tensor)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        TchOps::expand(tensor, shape)
    }

    fn int_sort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        TchOps::sort(tensor, dim, descending)
    }

    fn int_argsort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        TchOps::argsort(tensor, dim, descending)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_and(lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_or(lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_xor(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_not(tensor)
    }

    fn bitwise_and_scalar(
        lhs: IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> IntTensor<Self> {
        TchOps::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or_scalar(
        lhs: IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> IntTensor<Self> {
        TchOps::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor_scalar(
        lhs: IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> IntTensor<Self> {
        TchOps::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        TchOps::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(
        lhs: IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> IntTensor<Self> {
        TchOps::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(
        lhs: IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> IntTensor<Self> {
        TchOps::bitwise_right_shift_scalar(lhs, rhs)
    }
}
