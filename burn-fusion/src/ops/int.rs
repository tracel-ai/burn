use crate::FusionBackend;
use burn_tensor::{backend::Backend, ops::IntTensorOps};

impl<B: Backend> IntTensorOps<FusionBackend<B>> for FusionBackend<B> {
    fn int_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_shape<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn int_into_data<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<<FusionBackend<B> as Backend>::IntElem, D>> {
        todo!()
    }

    fn int_from_data<const D: usize>(
        data: burn_tensor::Data<<FusionBackend<B> as Backend>::IntElem, D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_device<const D: usize>(
        tensor: &<FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::Device {
        todo!()
    }

    fn int_to_device<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D2> {
        todo!()
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
        value: <FusionBackend<B> as Backend>::IntTensorPrimitive<D1>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_into_float<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn int_mask_where<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        mask: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        source: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mask_fill<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        mask: <FusionBackend<B> as Backend>::BoolTensorPrimitive<D>,
        value: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        value: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_select<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<1>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_select_assign<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
        indices: <FusionBackend<B> as Backend>::IntTensorPrimitive<1>,
        value: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_cat<const D: usize>(
        tensors: Vec<<FusionBackend<B> as Backend>::IntTensorPrimitive<D>>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_add<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_add_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div_scalar<const D: usize>(
        lhs: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        rhs: <FusionBackend<B> as Backend>::IntElem,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_zeros<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_ones<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<FusionBackend<B> as Backend>::Device,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sum<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_sum_dim<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mean_dim<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmax<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmin<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_abs<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_swap_dims<const D: usize>(
        tensor: <FusionBackend<B> as Backend>::IntTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <FusionBackend<B> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
