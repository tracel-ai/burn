use crate::{checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor, Autodiff};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, BoolTensorOps, IntTensor},
    Data, Device, Reader, Shape,
};

impl<B: Backend, C: CheckpointStrategy> BoolTensorOps<Self> for Autodiff<B, C> {
    fn bool_from_data<const D: usize>(data: Data<bool, D>, device: &Device<B>) -> BoolTensor<B, D> {
        B::bool_from_data(data, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> Reader<Data<bool, D>> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> Reader<Data<bool, D>> {
        B::bool_into_data(tensor)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D> {
        B::bool_into_int(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &Device<B>,
    ) -> BoolTensor<B, D> {
        B::bool_to_device(tensor, device)
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<B, D>) -> Device<B> {
        B::bool_device(tensor)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<B, D1> {
        B::bool_slice(tensor, ranges)
    }

    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> BoolTensor<B, D> {
        B::bool_empty(shape, device)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(tensors: Vec<BoolTensor<B, D>>, dim: usize) -> BoolTensor<B, D> {
        B::bool_cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<B, D>,
        rhs: BoolTensor<B, D>,
    ) -> BoolTensor<B, D> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_not<const D: usize>(tensor: BoolTensor<B, D>) -> BoolTensor<B, D> {
        B::bool_not(tensor)
    }

    fn bool_into_float<const D: usize>(
        tensor: BoolTensor<B, D>,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive<D> {
        AutodiffTensor::new(B::bool_into_float(tensor))
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <Autodiff<B> as Backend>::BoolTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <Autodiff<B> as Backend>::BoolTensorPrimitive<D> {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_narrow<const D: usize>(
        tensor: BoolTensor<B, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> BoolTensor<B, D> {
        B::bool_narrow(tensor, dim, start, length)
    }

    fn bool_chunk<const D: usize>(
        tensor: BoolTensor<B, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<BoolTensor<B, D>> {
        B::bool_chunk(tensor, chunks, dim)
    }

    fn bool_permute<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> BoolTensor<Self, D> {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip<const D: usize>(tensor: BoolTensor<B, D>, axes: &[usize]) -> BoolTensor<B, D> {
        B::bool_flip(tensor, axes)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn bool_argwhere<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, 2> {
        B::bool_argwhere(tensor)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn bool_nonzero<const D: usize>(tensor: BoolTensor<B, D>) -> Vec<IntTensor<B, 1>> {
        B::bool_nonzero(tensor)
    }

    fn bool_expand<const D: usize, const D2: usize>(
        tensor: BoolTensor<B, D>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2> {
        B::bool_expand(tensor, shape)
    }

    fn bool_repeat<const D: usize>(
        tensor: BoolTensor<B, D>,
        dim: usize,
        times: usize,
    ) -> BoolTensor<B, D> {
        B::bool_repeat(tensor, dim, times)
    }
}
