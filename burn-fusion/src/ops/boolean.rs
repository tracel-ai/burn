use crate::FusionBackend;
use burn_tensor::{backend::Backend, ops::BoolTensorOps};

impl<B: Backend> BoolTensorOps<FusionBackend<B>> for FusionBackend<B> {
    fn bool_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_shape<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        todo!()
    }

    fn bool_from_data<const D: usize>(
        data: burn_tensor::Data<bool, D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn bool_into_float<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn bool_device<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::Device {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D2> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<<FusionBackend<B> as Backend>::BoolTensorPrimitive<D>>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_not<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }
}
