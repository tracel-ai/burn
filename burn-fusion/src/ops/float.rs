use crate::FusionBackend;
use burn_tensor::{backend::Backend, ops::TensorOps};

impl<B: Backend> TensorOps<FusionBackend<B>> for FusionBackend<B> {
    fn from_data<const D: usize>(
        data: burn_tensor::Data<<FusionBackend<B> as Backend>::FloatElem, D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn random<const D: usize>(
        shape: burn_tensor::Shape<D>,
        distribution: burn_tensor::Distribution<<FusionBackend<B> as Backend>::FloatElem>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn shape<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn into_data<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<<FusionBackend<B> as Backend>::FloatElem, D>> {
        todo!()
    }

    fn device<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::Device {
        todo!()
    }

    fn to_device<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn into_int<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn add<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn add_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sub<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sub_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn matmul<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D2> {
        todo!()
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        value: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn select<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<1>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn select_assign<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<1>,
        value: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <FusionBackend<B> as Backend>::TensorPrimitive<D1>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn mask_where<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        mask: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        value: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        mask: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        value: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::FloatElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn sum<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn sum_dim<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mean_dim<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <<FusionBackend<B> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        tensor: <<FusionBackend<B> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<
            D,
        >,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log1p<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn powf<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        value: f32,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sqrt<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn abs<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cos<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sin<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn tanh<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn erf<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cat<const D: usize>(
        tensors: Vec<<FusionBackend<B> as Backend>::TensorPrimitive<D>>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn argmax<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
