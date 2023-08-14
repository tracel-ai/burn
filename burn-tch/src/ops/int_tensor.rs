use std::ops::Range;

use burn_tensor::{backend::Backend, ops::IntTensorOps, Data, Shape};

use crate::{element::TchElement, TchBackend, TchDevice, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement> IntTensorOps<TchBackend<E>> for TchBackend<E> {
    fn int_from_data<const D: usize>(data: Data<i64, D>, device: &TchDevice) -> TchTensor<i64, D> {
        TchTensor::from_data(data, (*device).into())
    }

    fn int_shape<const D: usize>(tensor: &TchTensor<i64, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_repeat<const D: usize>(
        tensor: TchTensor<i64, D>,
        dim: usize,
        times: usize,
    ) -> TchTensor<i64, D> {
        TchOps::repeat(tensor, dim, times)
    }

    fn int_to_data<const D: usize>(tensor: &TchTensor<i64, D>) -> Data<i64, D> {
        let shape = Self::int_shape(tensor);
        let tensor = Self::int_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<i64>, tch::TchError> = tensor.tensor.shallow_clone().try_into();

        Data::new(values.unwrap(), shape)
    }

    fn int_into_data<const D: usize>(tensor: TchTensor<i64, D>) -> Data<i64, D> {
        Self::int_to_data(&tensor)
    }

    fn int_to_device<const D: usize>(
        tensor: TchTensor<i64, D>,
        device: &TchDevice,
    ) -> TchTensor<i64, D> {
        TchTensor::new(tensor.tensor.to((*device).into()))
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<i64, D2> {
        TchOps::reshape(tensor, shape)
    }

    fn int_device<const D: usize>(tensor: &TchTensor<i64, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let tensor = tch::Tensor::empty(
            shape.dims.map(|a| a as i64),
            (tch::Kind::Int64, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        ranges: [Range<usize>; D2],
    ) -> TchTensor<i64, D1> {
        TchOps::slice(tensor, ranges)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: TchTensor<i64, D1>,
    ) -> TchTensor<i64, D1> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat<const D: usize>(tensors: Vec<TchTensor<i64, D>>, dim: usize) -> TchTensor<i64, D> {
        TchOps::cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::equal_elem(lhs, rhs)
    }

    fn int_greater<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::greater_elem(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: i64,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal_elem(lhs, rhs)
    }

    fn int_lower<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::lower_elem(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: i64,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal_elem(lhs, rhs)
    }

    fn int_add<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn int_sub<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn int_mul<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn int_div<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        )
    }

    fn int_neg<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, D> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_ones<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_full<const D: usize>(
        shape: Shape<D>,
        fill_value: i64,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::full(
            shape.dims,
            fill_value,
            (tch::Kind::Int64, device),
        ))
    }

    fn int_sum<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, 1> {
        TchOps::sum(tensor)
    }

    fn int_sum_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::sum_dim(tensor, dim)
    }

    fn int_mean<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, 1> {
        TchOps::mean(tensor)
    }

    fn int_mean_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::mean_dim(tensor, dim)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: TchTensor<i64, D>,
        indices: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: TchTensor<i64, D>,
        indices: TchTensor<i64, D>,
        value: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: TchTensor<i64, D>,
        dim: usize,
        indices: TchTensor<i64, 1>,
    ) -> TchTensor<i64, D> {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: TchTensor<i64, D>,
        dim: usize,
        indices: TchTensor<i64, 1>,
        value: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn int_mask_where<const D: usize>(
        tensor: TchTensor<i64, D>,
        mask: TchTensor<bool, D>,
        source: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchTensor::binary_ops_tensor(
            tensor,
            source,
            |tensor, source| tensor.f_masked_scatter_(&mask.tensor, source).unwrap(),
            |tensor, source| tensor.f_masked_scatter(&mask.tensor, source).unwrap(),
            |tensor, source| tensor.f_masked_scatter(&mask.tensor, source).unwrap(),
        )
    }

    fn int_mask_fill<const D: usize>(
        tensor: TchTensor<i64, D>,
        mask: TchTensor<bool, D>,
        value: i64,
    ) -> TchTensor<i64, D> {
        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn int_argmax<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::argmax(tensor, dim)
    }

    fn int_argmin<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::argmin(tensor, dim)
    }

    fn int_max_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices<const D: usize>(
        tensor: TchTensor<i64, D>,
        dim: usize,
    ) -> (TchTensor<i64, D>, TchTensor<i64, D>) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices<const D: usize>(
        tensor: TchTensor<i64, D>,
        dim: usize,
    ) -> (TchTensor<i64, D>, TchTensor<i64, D>) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn int_clamp_min<const D: usize>(tensor: TchTensor<i64, D>, min: i64) -> TchTensor<i64, D> {
        TchOps::clamp_min(tensor, min)
    }

    fn int_clamp_max<const D: usize>(tensor: TchTensor<i64, D>, max: i64) -> TchTensor<i64, D> {
        TchOps::clamp_max(tensor, max)
    }

    fn int_clamp<const D: usize>(
        tensor: TchTensor<i64, D>,
        min: i64,
        max: i64,
    ) -> TchTensor<i64, D> {
        TchOps::clamp(tensor, min, max)
    }

    fn int_abs<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, D> {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn int_into_float<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<E, D> {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }
}
