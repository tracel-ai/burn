use crate::{FusedBackend, Fusion};
use burn_tensor::{
    ops::{BoolTensor, BoolTensorOps},
    Device, Shape,
};

impl<B: FusedBackend> BoolTensorOps<Self> for Fusion<B> {
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        tensor.bool_into_data()
    }

    fn bool_from_data<const D: usize>(
        data: burn_tensor::Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::IntTensor<Self, D> {
        todo!()
    }

    fn bool_into_float<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        todo!()
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<Self, D>) -> Device<Self> {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<BoolTensor<Self, D>>,
        dim: usize,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_not<const D: usize>(tensor: BoolTensor<Self, D>) -> BoolTensor<Self, D> {
        todo!()
    }

    fn bool_swap_dims<const D: usize>(
        tensor: BoolTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<Self, D> {
        todo!()
    }
}
