use burn_tensor::ops::TensorOps;

use crate::{element::CandleElement, CandleBackend};

impl<E: CandleElement> TensorOps<CandleBackend<E>> for CandleBackend<E> {
    fn from_data<const D: usize>(
        data: burn_tensor::Data<<CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem, D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn random<const D: usize>(
        shape: burn_tensor::Shape<D>,
        distribution: burn_tensor::Distribution<
            <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
        >,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn shape<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn to_data<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> burn_tensor::Data<<CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem, D> {
        todo!()
    }

    fn device<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::Device {
        todo!()
    }

    fn to_device<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn into_int<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<CandleBackend<E> as burn_tensor::backend::Backend>::Device,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn add<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn add_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sub<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sub_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div_scalar<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn matmul<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D2> {
        todo!()
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn select<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<1>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn select_assign<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
        indices: <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<1>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn mask_where<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        mask: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        mask: <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        value: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal_elem<const D: usize>(
        lhs: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        rhs: <CandleBackend<E> as burn_tensor::backend::Backend>::FloatElem,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn sum<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn sum_dim<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mean_dim<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        tensor: &<CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <<CandleBackend<E> as burn_tensor::backend::Backend>::FullPrecisionBackend as burn_tensor::backend::Backend>::TensorPrimitive<D>{
        todo!()
    }

    fn from_full_precision<const D: usize>(
        tensor: <<CandleBackend<E> as burn_tensor::backend::Backend>::FullPrecisionBackend as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log1p<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn powf<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        value: f32,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sqrt<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn abs<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cos<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sin<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn tanh<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn erf<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cat<const D: usize>(
        tensors: Vec<<CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn argmax<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(
        tensor: <CandleBackend<E> as burn_tensor::backend::Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <CandleBackend<E> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
