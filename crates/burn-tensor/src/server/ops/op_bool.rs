use crate::{
    ops::BoolTensorOps,
    server::{Server, ServerBackend},
};

impl<B: ServerBackend> BoolTensorOps<Self> for Server<B> {
    fn bool_empty<const D: usize>(
        shape: crate::Shape<D>,
        device: &crate::Device<Self>,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(tensor: &crate::ops::BoolTensor<Self, D>) -> crate::Shape<D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
    ) -> impl core::future::Future<Output = crate::TensorData> + Send {
        async { tensor.into_data().await }
    }

    fn bool_from_data<const D: usize>(
        data: crate::TensorData,
        device: &crate::Device<Self>,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
    ) -> crate::ops::IntTensor<Self, D> {
        todo!()
    }

    fn bool_into_float<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
    ) -> crate::ops::FloatTensor<Self, D> {
        todo!()
    }

    fn bool_device<const D: usize>(
        tensor: &crate::ops::BoolTensor<Self, D>,
    ) -> crate::Device<Self> {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
        device: &crate::Device<Self>,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: crate::ops::BoolTensor<Self, D1>,
        shape: crate::Shape<D2>,
    ) -> crate::ops::BoolTensor<Self, D2> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: crate::ops::BoolTensor<Self, D1>,
        ranges: [core::ops::Range<usize>; D2],
    ) -> crate::ops::BoolTensor<Self, D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: crate::ops::BoolTensor<Self, D1>,
        ranges: [core::ops::Range<usize>; D2],
        value: crate::ops::BoolTensor<Self, D1>,
    ) -> crate::ops::BoolTensor<Self, D1> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: crate::ops::BoolTensor<Self, D>,
        rhs: crate::ops::BoolTensor<Self, D>,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_not<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_swap_dims<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_permute<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_flip<const D: usize>(
        tensor: crate::ops::BoolTensor<Self, D>,
        axes: &[usize],
    ) -> crate::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: crate::ops::BoolTensor<Self, D1>,
        shape: crate::Shape<D2>,
    ) -> crate::ops::BoolTensor<Self, D2> {
        todo!()
    }
}
