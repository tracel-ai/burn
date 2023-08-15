use burn_tensor::ops::IntTensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> IntTensorOps<CandleBackend<E>> for CandleBackend<E> {
    fn int_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_shape<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn int_into_data<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Data<<CandleBackend<E> as burn_tensor::backend::Backend>::IntElem, D> {
        todo!()
    }

    fn int_from_data<const D: usize>(
        data: burn_tensor::Data<<CandleBackend<E> as burn_tensor::backend::Backend>::IntElem, D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_device<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::Device {
        todo!()
    }

    fn int_to_device<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D2> {
        todo!()
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_into_float<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn int_mask_where<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        mask: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        source: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mask_fill<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        mask: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_select<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<1>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_select_assign<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<1>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_cat<const D: usize>(
        tensors: Vec<<CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_add<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_add_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::IntElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_zeros<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_ones<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sum<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_sum_dim<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mean_dim<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmax<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmin<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_abs<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
