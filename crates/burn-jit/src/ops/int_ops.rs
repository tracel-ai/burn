use super::{expand, numeric, permute};
use crate::codegen::dialect::gpu::{Elem, Item, Operator, Scope, UnaryOperator};
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::{kernel, unary, JitBackend, Runtime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};
use burn_tensor::{ops::IntTensorOps, Data, Distribution, ElementConversion, Reader, Shape};
use std::ops::Range;

impl<R: Runtime> IntTensorOps<Self> for JitBackend<R> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        super::empty(shape, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        super::into_data(tensor)
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::from_data(data, device)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::to_device(tensor, device)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::mask_where_auto(tensor, mask, value)
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        kernel::mask_fill_auto(tensor, mask, value)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        kernel::select(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::equal_elem(lhs, rhs)
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_elem(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_elem(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::add_scalar(lhs, rhs)
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::div_scalar(lhs, rhs)
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        numeric::zeros(shape, device)
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        numeric::ones(shape, device)
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        kernel::reduce::sum(tensor, Default::default())
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::sum_dim(tensor, dim, Default::default())
    }

    fn int_prod<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        kernel::reduce::prod(tensor, Default::default())
    }

    fn int_prod_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::prod_dim(tensor, dim, Default::default())
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::mean_dim(tensor, dim, Default::default())
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::argmax(tensor, dim, Default::default())
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::argmin(tensor, dim, Default::default())
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        kernel::clamp(tensor, min, max)
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary!(
            operation: |scope: &mut Scope, elem: Elem| Operator::Abs(UnaryOperator {
                input: scope.read_array(0, Item::Scalar(elem)),
                out: scope.create_local(elem),
            }),
            runtime: R,
            input: tensor,
            elem: IntElem<Self>
        )
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn int_swap_dims<const D: usize>(
        mut tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn int_repeat<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> IntTensor<Self, D> {
        kernel::repeat(tensor, dim, times)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let float_tensor = match distribution {
            Distribution::Default => random_uniform(shape, device, 0.elem::<f32>(), 255.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem(), high.elem())
            }
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob.elem()),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem(), std.elem())
            }
        };

        kernel::cast(float_tensor)
    }

    fn int_permute<const D: usize>(
        tensor: IntTensor<Self, D>,
        axes: [usize; D],
    ) -> IntTensor<Self, D> {
        permute(tensor, axes)
    }

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        expand(tensor, shape)
    }

    fn int_flip<const D: usize>(tensor: IntTensor<Self, D>, axes: &[usize]) -> IntTensor<Self, D> {
        kernel::flip(tensor, axes)
    }
}
