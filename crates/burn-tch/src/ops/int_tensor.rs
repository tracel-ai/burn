use std::ops::Range;

use burn_tensor::{
    backend::Backend,
    ops::{IntTensor, IntTensorOps},
    Distribution, Shape, TensorData,
};

use crate::{element::TchElement, LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement, Q: QuantElement> IntTensorOps<Self> for LibTorch<E, Q> {
    fn int_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor<i64> {
        TchTensor::from_data(data, (*device).into())
    }

    fn int_shape(tensor: &TchTensor<i64>) -> Shape {
        tensor.shape()
    }

    fn int_repeat_dim(tensor: TchTensor<i64>, dim: usize, times: usize) -> TchTensor<i64> {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn int_into_data(tensor: TchTensor<i64>) -> TensorData {
        let shape = Self::int_shape(&tensor);
        let tensor = Self::int_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<i64>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        TensorData::new(values.unwrap(), shape)
    }

    fn int_to_device(tensor: TchTensor<i64>, device: &LibTorchDevice) -> TchTensor<i64> {
        TchOps::to_device(tensor, device)
    }

    fn int_reshape(tensor: TchTensor<i64>, shape: Shape) -> TchTensor<i64> {
        TchOps::reshape(tensor, shape)
    }

    fn int_device(tensor: &TchTensor<i64>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn int_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor<i64> {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (tch::Kind::Int64, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn int_slice(tensor: TchTensor<i64>, ranges: &[Range<usize>]) -> TchTensor<i64> {
        TchOps::slice(tensor, ranges)
    }

    fn int_slice_assign(
        tensor: TchTensor<i64>,
        ranges: &[Range<usize>],
        value: TchTensor<i64>,
    ) -> TchTensor<i64> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<TchTensor<i64>>, dim: usize) -> TchTensor<i64> {
        TchOps::cat(tensors, dim)
    }

    fn int_equal(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<bool> {
        TchOps::equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<bool> {
        TchOps::equal_elem(lhs, rhs)
    }

    fn int_greater(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<bool> {
        TchOps::greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<bool> {
        TchOps::greater_elem(lhs, rhs)
    }

    fn int_greater_equal(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<bool> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<bool> {
        TchOps::greater_equal_elem(lhs, rhs)
    }

    fn int_lower(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<bool> {
        TchOps::lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<bool> {
        TchOps::lower_elem(lhs, rhs)
    }

    fn int_lower_equal(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<bool> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<bool> {
        TchOps::lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::add(lhs, rhs)
    }

    fn int_add_scalar(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<i64> {
        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn int_sub(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<i64> {
        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn int_mul(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<i64> {
        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn int_div(lhs: TchTensor<i64>, rhs: TchTensor<i64>) -> TchTensor<i64> {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor<f64> =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));
        let rhs: TchTensor<f64> =
            TchTensor::new(rhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out = TchOps::div(lhs, rhs);

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn int_div_scalar(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<i64> {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor<f64> =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out: TchTensor<f64> = lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        );

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn int_remainder_scalar(lhs: TchTensor<i64>, rhs: i64) -> TchTensor<i64> {
        lhs.unary_ops(
            |tensor| tensor.f_remainder(rhs).unwrap(),
            |tensor| tensor.f_remainder(rhs).unwrap(),
        )
    }

    fn int_neg(tensor: TchTensor<i64>) -> TchTensor<i64> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor<i64> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_ones(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor<i64> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_full(
        shape: Shape,
        fill_value: i64,
        device: &<LibTorch<E> as Backend>::Device,
    ) -> TchTensor<i64> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::full(
            shape.dims,
            fill_value,
            (tch::Kind::Int64, device),
        ))
    }

    fn int_sum(tensor: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::sum(tensor)
    }

    fn int_sum_dim(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::prod(tensor)
    }

    fn int_prod_dim(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::prod_dim(tensor, dim)
    }

    fn int_mean(tensor: TchTensor<i64>) -> TchTensor<i64> {
        let tensor: TchTensor<f64> =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));
        let output: TchTensor<i64> = TchTensor::new(TchOps::mean(tensor).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn int_mean_dim(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        let tensor: TchTensor<f64> =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));

        let output: TchTensor<i64> = TchTensor::new(TchOps::mean_dim(tensor, dim).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn int_gather(dim: usize, tensor: TchTensor<i64>, indices: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: TchTensor<i64>,
        indices: TchTensor<i64>,
        value: TchTensor<i64>,
    ) -> TchTensor<i64> {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn int_select(tensor: TchTensor<i64>, dim: usize, indices: TchTensor<i64>) -> TchTensor<i64> {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: TchTensor<i64>,
        dim: usize,
        indices: TchTensor<i64>,
        value: TchTensor<i64>,
    ) -> TchTensor<i64> {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn int_mask_where(
        tensor: TchTensor<i64>,
        mask: TchTensor<bool>,
        source: TchTensor<i64>,
    ) -> TchTensor<i64> {
        TchTensor::binary_ops_tensor(
            tensor,
            source,
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
        )
    }

    fn int_mask_fill(tensor: TchTensor<i64>, mask: TchTensor<bool>, value: i64) -> TchTensor<i64> {
        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn int_argmax(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::argmax(tensor, dim)
    }

    fn int_argmin(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::argmin(tensor, dim)
    }

    fn int_max_dim(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: TchTensor<i64>,
        dim: usize,
    ) -> (TchTensor<i64>, TchTensor<i64>) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim(tensor: TchTensor<i64>, dim: usize) -> TchTensor<i64> {
        TchOps::min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: TchTensor<i64>,
        dim: usize,
    ) -> (TchTensor<i64>, TchTensor<i64>) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn int_clamp_min(tensor: TchTensor<i64>, min: i64) -> TchTensor<i64> {
        TchOps::clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: TchTensor<i64>, max: i64) -> TchTensor<i64> {
        TchOps::clamp_max(tensor, max)
    }

    fn int_clamp(tensor: TchTensor<i64>, min: i64, max: i64) -> TchTensor<i64> {
        TchOps::clamp(tensor, min, max)
    }

    fn int_abs(tensor: TchTensor<i64>) -> TchTensor<i64> {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn int_into_float(tensor: TchTensor<i64>) -> TchTensor<E> {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow(
        tensor: TchTensor<i64>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<i64> {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn int_chunk(tensor: TchTensor<i64>, chunks: usize, dim: usize) -> Vec<TchTensor<i64>> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &LibTorchDevice,
    ) -> TchTensor<i64> {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::<i64>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.uniform_(0.0, 255.0))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<i64>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<i64>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::<i64>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn int_arange(range: Range<i64>, device: &LibTorchDevice) -> TchTensor<i64> {
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
}
