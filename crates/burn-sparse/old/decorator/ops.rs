use crate::decorator::SparseDecorator;
use crate::decorator::SparseRepresentation;
use burn_tensor::{
    backend::Backend,
    ops::{
        ActivationOps, BoolTensor, BoolTensorOps, ConvOptions, ConvTransposeOptions, FloatTensor,
        FloatTensorOps, IntElem, IntTensor, IntTensorOps, InterpolateOptions, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps, QTensorOps,
    },
    Device, Distribution, Shape, TensorData,
};
use core::ops::Range;

impl<B, R> FloatTensorOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
    fn float_random<const D: usize>(
        shape: burn_tensor::Shape<D>,
        distribution: burn_tensor::Distribution,
        device: &burn_tensor::Device<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_random(shape, distribution, device)
    }

    fn float_shape<const D: usize>(
        tensor: &burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::Shape<D> {
        B::float_shape(tensor)
    }

    fn float_device<const D: usize>(
        tensor: &burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::Device<B> {
        B::float_device(tensor)
    }

    fn float_to_device<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        device: &burn_tensor::Device<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_to_device(tensor, device)
    }

    fn float_into_int<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::IntTensor<B, D> {
        B::float_into_int(tensor)
    }

    fn float_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_empty(shape, device)
    }

    fn float_add<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_add(lhs, rhs)
    }

    fn float_add_scalar<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_add_scalar(lhs, rhs)
    }

    fn float_sub<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_sub(lhs, rhs)
    }

    fn float_sub_scalar<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_sub_scalar(lhs, rhs)
    }

    fn float_mul<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_mul(lhs, rhs)
    }

    fn float_mul_scalar<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_mul_scalar(lhs, rhs)
    }

    fn float_div<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_div(lhs, rhs)
    }

    fn float_div_scalar<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_div_scalar(lhs, rhs)
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_matmul(lhs, rhs)
    }

    fn float_recip<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_recip(tensor)
    }

    fn float_swap_dims<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_permute<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        axes: [usize; D],
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_permute(tensor, axes)
    }

    fn float_flip<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        axes: &[usize],
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_flip(tensor, axes)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> burn_tensor::ops::FloatTensor<B, D2> {
        B::float_reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_gather(dim, tensor, indices)
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        indices: burn_tensor::ops::IntTensor<B, D>,
        value: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_scatter(dim, tensor, indices, value)
    }

    fn float_select<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_select(tensor, dim, indices)
    }

    fn float_select_assign<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
        indices: burn_tensor::ops::IntTensor<B, 1>,
        value: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_select_assign(tensor, dim, indices, value)
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D1>,
        ranges: [core::ops::Range<usize>; D2],
    ) -> burn_tensor::ops::FloatTensor<B, D1> {
        B::float_slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D1>,
        ranges: [core::ops::Range<usize>; D2],
        value: burn_tensor::ops::FloatTensor<B, D1>,
    ) -> burn_tensor::ops::FloatTensor<B, D1> {
        B::float_slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        mask: burn_tensor::ops::BoolTensor<B, D>,
        value: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        mask: burn_tensor::ops::BoolTensor<B, D>,
        value: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_mask_fill(tensor, mask, value)
    }

    fn float_equal<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_equal(lhs, rhs)
    }

    fn float_equal_elem<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_equal_elem(lhs, rhs)
    }

    fn float_greater<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_greater(lhs, rhs)
    }

    fn float_greater_elem<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_greater_elem(lhs, rhs)
    }

    fn float_greater_equal<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_greater_equal_elem(lhs, rhs)
    }

    fn float_lower<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_lower(lhs, rhs)
    }

    fn float_lower_elem<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_lower_elem(lhs, rhs)
    }

    fn float_lower_equal<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatElem<B>,
    ) -> burn_tensor::ops::BoolTensor<B, D> {
        B::float_lower_equal_elem(lhs, rhs)
    }

    fn float_sum<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, 1> {
        B::float_sum(tensor)
    }

    fn float_sum_dim<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_sum_dim(tensor, dim)
    }

    fn float_mean_dim<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_mean_dim(tensor, dim)
    }

    fn float_exp<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_exp(tensor)
    }

    fn float_log<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_log(tensor)
    }

    fn float_log1p<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_log1p(tensor)
    }

    fn float_powf<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<B, D>,
        rhs: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_powf(lhs, rhs)
    }

    fn float_powf_scalar<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        value: f32,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_powf_scalar(tensor, value)
    }

    fn float_sqrt<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_sqrt(tensor)
    }

    fn float_abs<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_abs(tensor)
    }

    fn float_cos<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_cos(tensor)
    }

    fn float_sin<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_sin(tensor)
    }

    fn float_tanh<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_tanh(tensor)
    }

    fn float_erf<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
    ) -> burn_tensor::ops::FloatTensor<B, D> {
        B::float_erf(tensor)
    }

    fn float_argmax<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
    ) -> burn_tensor::ops::IntTensor<B, D> {
        B::float_argmax(tensor, dim)
    }

    fn float_argmin<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D>,
        dim: usize,
    ) -> burn_tensor::ops::IntTensor<B, D> {
        B::float_argmin(tensor, dim)
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<B, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> burn_tensor::ops::FloatTensor<B, D2> {
        B::float_expand(tensor, shape)
    }

    fn float_into_data<const D: usize>(
        tensor: FloatTensor<SparseDecorator<B, R>, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        B::float_into_data(tensor)
    }

    fn float_from_data<const D: usize>(
        data: TensorData,
        device: &Device<SparseDecorator<B, R>>,
    ) -> FloatTensor<SparseDecorator<B, R>, D> {
        B::float_from_data(data, device)
    }
}

impl<B, R> BoolTensorOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
    fn bool_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<SparseDecorator<B, R>>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_empty(shape, device)
    }

    fn bool_shape<const D: usize>(
        tensor: &burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_into_int<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::ops::IntTensor<SparseDecorator<B, R>, D> {
        B::bool_into_int(tensor)
    }

    fn bool_into_float<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::ops::FloatTensor<SparseDecorator<B, R>, D> {
        B::bool_into_float(tensor)
    }

    fn bool_device<const D: usize>(
        tensor: &burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::Device<SparseDecorator<B, R>> {
        B::bool_device(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
        device: &burn_tensor::Device<SparseDecorator<B, R>>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1>,
        ranges: [core::ops::Range<usize>; D2],
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1> {
        B::bool_slice(tensor, ranges)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1>,
        ranges: [core::ops::Range<usize>; D2],
        value: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1> {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn bool_equal<const D: usize>(
        lhs: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
        rhs: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_not<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_not(tensor)
    }

    fn bool_swap_dims<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
        dim1: usize,
        dim2: usize,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
        axes: [usize; D],
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D>,
        axes: &[usize],
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_flip(tensor, axes)
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> burn_tensor::ops::BoolTensor<SparseDecorator<B, R>, D2> {
        B::bool_expand(tensor, shape)
    }

    fn bool_into_data<const D: usize>(
        tensor: BoolTensor<SparseDecorator<B, R>, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        B::bool_into_data(tensor)
    }

    fn bool_from_data<const D: usize>(
        data: TensorData,
        device: &Device<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::bool_from_data(data, device)
    }
}

impl<B, R> IntTensorOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_empty(shape, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<SparseDecorator<B, R>, D>) -> Shape<D> {
        B::int_shape(tensor)
    }

    fn int_device<const D: usize>(
        tensor: &IntTensor<SparseDecorator<B, R>, D>,
    ) -> Device<SparseDecorator<B, R>> {
        B::int_device(tensor)
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_to_device(tensor, device)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<SparseDecorator<B, R>, D2> {
        B::int_reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D1>,
        indices: [Range<usize>; D2],
    ) -> IntTensor<SparseDecorator<B, R>, D1> {
        B::int_slice(tensor, indices)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D1>,
        indices: [Range<usize>; D2],
        value: IntTensor<SparseDecorator<B, R>, D1>,
    ) -> IntTensor<SparseDecorator<B, R>, D1> {
        B::int_slice_assign(tensor, indices, value)
    }

    fn int_into_float<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> FloatTensor<SparseDecorator<B, R>, D> {
        B::int_into_float(tensor)
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        mask: BoolTensor<SparseDecorator<B, R>, D>,
        source: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_mask_where(tensor, mask, source)
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        mask: BoolTensor<SparseDecorator<B, R>, D>,
        value: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_mask_fill(tensor, mask, value)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        indices: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        indices: IntTensor<SparseDecorator<B, R>, D>,
        value: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_scatter(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
        indices: IntTensor<SparseDecorator<B, R>, 1>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_select(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
        indices: IntTensor<SparseDecorator<B, R>, 1>,
        value: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_select_assign(tensor, dim, indices, value)
    }

    fn int_cat<const D: usize>(
        tensors: Vec<IntTensor<SparseDecorator<B, R>, D>>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_equal_elem(lhs, rhs)
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_greater(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_lower(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> BoolTensor<SparseDecorator<B, R>, D> {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_sub_scalar(lhs, rhs)
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_mul_scalar(lhs, rhs)
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_div_scalar(lhs, rhs)
    }

    fn int_remainder_scalar<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_remainder_scalar(lhs, rhs)
    }

    fn int_zeros<const D: usize>(
        shape: Shape<D>,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_zeros(shape, device)
    }

    fn int_ones<const D: usize>(
        shape: Shape<D>,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_ones(shape, device)
    }

    fn int_sum<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, 1> {
        B::int_sum(tensor)
    }

    fn int_sum_dim<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_sum_dim(tensor, dim)
    }

    fn int_prod<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, 1> {
        B::int_prod(tensor)
    }

    fn int_prod_dim<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_prod_dim(tensor, dim)
    }

    fn int_mean_dim<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_mean_dim(tensor, dim)
    }

    fn int_argmax<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_argmax(tensor, dim)
    }

    fn int_argmin<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_argmin(tensor, dim)
    }

    fn int_max_dim<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> (
        IntTensor<SparseDecorator<B, R>, D>,
        IntTensor<SparseDecorator<B, R>, D>,
    ) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
    ) -> (
        IntTensor<SparseDecorator<B, R>, D>,
        IntTensor<SparseDecorator<B, R>, D>,
    ) {
        B::int_min_dim_with_indices(tensor, dim)
    }

    fn int_abs<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_abs(tensor)
    }

    fn int_transpose<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_transpose(tensor)
    }

    fn int_swap_dims<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_permute<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        axes: [usize; D],
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_permute(tensor, axes)
    }

    fn int_flip<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        axes: &[usize],
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_flip(tensor, axes)
    }

    fn int_narrow<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_narrow(tensor, dim, start, length)
    }

    fn int_chunk<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<IntTensor<SparseDecorator<B, R>, D>> {
        B::int_chunk(tensor, chunks, dim)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_random(shape, distribution, device)
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntTensor<SparseDecorator<B, R>, D>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<SparseDecorator<B, R>, D>,
        rhs: IntElem<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_add_scalar(lhs, rhs)
    }

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<SparseDecorator<B, R>, D2> {
        B::int_expand(tensor, shape)
    }

    fn int_into_data<const D: usize>(
        tensor: IntTensor<SparseDecorator<B, R>, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        B::int_into_data(tensor)
    }

    fn int_from_data<const D: usize>(
        data: TensorData,
        device: &Device<SparseDecorator<B, R>>,
    ) -> IntTensor<SparseDecorator<B, R>, D> {
        B::int_from_data(data, device)
    }
}

impl<B, R> QTensorOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
    fn q_shape<const D: usize>(tensor: &burn_tensor::ops::QuantizedTensor<B, D>) -> Shape<D> {
        B::q_shape(tensor)
    }

    fn q_device<const D: usize>(tensor: &burn_tensor::ops::QuantizedTensor<B, D>) -> Device<B> {
        B::q_device(tensor)
    }

    fn q_from_data<const D: usize>(
        data: TensorData,
        device: &Device<SparseDecorator<B, R>>,
    ) -> burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D> {
        B::q_from_data(data, device)
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D1>,
        shape: Shape<D2>,
    ) -> burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D2> {
        B::q_reshape(tensor, shape)
    }

    fn q_into_data<const D: usize>(
        tensor: burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D>,
    ) -> impl std::future::Future<Output = TensorData> + Send {
        B::q_into_data(tensor)
    }

    fn quantize<const D: usize>(
        tensor: FloatTensor<SparseDecorator<B, R>, D>,
        scheme: &burn_tensor::quantization::QuantizationScheme,
        qparams: burn_tensor::quantization::QuantizationParametersPrimitive<SparseDecorator<B, R>>,
    ) -> burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D> {
        B::quantize(tensor, scheme, qparams)
    }

    fn dequantize<const D: usize>(
        tensor: burn_tensor::ops::QuantizedTensor<SparseDecorator<B, R>, D>,
    ) -> FloatTensor<SparseDecorator<B, R>, D> {
        B::dequantize(tensor)
    }
}

impl<B, R> ModuleOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
    fn conv2d(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        weight: FloatTensor<SparseDecorator<B, R>, 4>,
        bias: Option<FloatTensor<SparseDecorator<B, R>, 1>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::conv2d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        weight: FloatTensor<SparseDecorator<B, R>, 4>,
        bias: Option<FloatTensor<SparseDecorator<B, R>, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::conv_transpose2d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        grad: FloatTensor<SparseDecorator<B, R>, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn max_pool2d(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::max_pool2d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<SparseDecorator<B, R>> {
        let MaxPool2dWithIndices { output, indices } =
            B::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation);
        MaxPool2dWithIndices { output, indices }
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<SparseDecorator<B, R>, 4>,
        indices: IntTensor<SparseDecorator<B, R>, 4>,
    ) -> MaxPool2dBackward<SparseDecorator<B, R>> {
        let MaxPool2dBackward { x_grad } = B::max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        );
        MaxPool2dBackward { x_grad }
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        output_size: [usize; 2],
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        grad: FloatTensor<SparseDecorator<B, R>, 4>,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::adaptive_avg_pool2d_backward(x, grad)
    }

    fn interpolate(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<SparseDecorator<B, R>, 4>,
        grad: FloatTensor<SparseDecorator<B, R>, 4>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<SparseDecorator<B, R>, 4> {
        B::interpolate_backward(x, grad, output_size, options)
    }

    fn conv3d(
        x: FloatTensor<B, 5>,
        weight: FloatTensor<B, 5>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<B, 5> {
        B::conv3d(x, weight, bias, options)
    }

    fn conv_transpose3d(
        x: FloatTensor<B, 5>,
        weight: FloatTensor<B, 5>,
        bias: Option<FloatTensor<B, 1>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<B, 5> {
        B::conv_transpose3d(x, weight, bias, options)
    }
}

impl<B, R> ActivationOps<SparseDecorator<B, R>> for SparseDecorator<B, R>
where
    B: Backend,
    R: SparseRepresentation,
{
}
