use burn_tensor::ops::BoolTensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> BoolTensorOps<CandleBackend<E>> for CandleBackend<E> {
    fn bool_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_shape<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Data<bool, D> {
        todo!()
    }

    fn bool_from_data<const D: usize>(
        data: burn_tensor::Data<bool, D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn bool_into_float<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn bool_device<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::Device {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D2> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<<CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        rhs: bool,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }
}
