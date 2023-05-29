use burn_tensor::{
    backend::Backend,
    ops::{BoolTensorOps, IntTensorOps, TensorOps},
};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend, WGPUDevice,
};

impl<G, F, I> IntTensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn int_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_shape<const D: usize>(
        tensor: &<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn int_into_data<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> burn_tensor::Data<<WGPUBackend<G, F, I> as Backend>::IntElem, D> {
        todo!()
    }

    fn int_from_data<const D: usize>(
        data: burn_tensor::Data<<WGPUBackend<G, F, I> as Backend>::IntElem, D>,
        device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_device<const D: usize>(
        tensor: &<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::Device {
        todo!()
    }

    fn int_to_device<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D2> {
        todo!()
    }

    fn int_index<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_index_assign<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_mask_scatter<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        source: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mask_fill<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        value: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_index_select_dim<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        dim: usize,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
        value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_cat<const D: usize>(
        tensors: Vec<<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_add<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_add_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sub_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mul_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_div_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_neg<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_zeros<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_ones<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sum<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_sum_dim<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mean<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_mean_dim<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmax<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmin<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
