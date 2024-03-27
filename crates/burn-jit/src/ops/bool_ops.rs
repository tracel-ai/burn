use crate::{kernel, JitBackend, Runtime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_tensor::Reader;
use burn_tensor::{ops::BoolTensorOps, Data, Shape};
use std::ops::Range;

use super::{expand, permute};

impl<R: Runtime> BoolTensorOps<Self> for JitBackend<R> {
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        super::empty(shape, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<Self, D>) -> Reader<Data<bool, D>> {
        super::bool_into_data(tensor)
    }

    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let data: Data<u32, D> = Data::new(
            data.value
                .into_iter()
                .map(|c| match c {
                    true => 1,
                    false => 0,
                })
                .collect(),
            data.shape,
        );
        super::from_data(data, device)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<Self, D>) -> IntTensor<Self, D> {
        kernel::bool_cast(tensor)
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        super::to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn bool_not<const D: usize>(tensor: BoolTensor<Self, D>) -> BoolTensor<Self, D> {
        kernel::equal_elem(tensor, 0)
    }

    fn bool_into_float<const D: usize>(tensor: BoolTensor<Self, D>) -> FloatTensor<Self, D> {
        kernel::bool_cast(tensor)
    }

    fn bool_swap_dims<const D: usize>(
        mut tensor: BoolTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<Self, D> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn bool_repeat<const D: usize>(
        tensor: BoolTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> BoolTensor<Self, D> {
        kernel::repeat(tensor, dim, times)
    }

    fn bool_permute<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> BoolTensor<Self, D> {
        permute(tensor, axes)
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        expand(tensor, shape)
    }

    fn bool_flip<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: &[usize],
    ) -> BoolTensor<Self, D> {
        kernel::flip(tensor, axes)
    }
}
