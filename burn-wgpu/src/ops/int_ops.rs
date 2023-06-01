use burn_tensor::{backend::Backend, ops::IntTensorOps, Data, Shape};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend,
};

use super::{numeric::NumericOps, BaseOps, Device, IntElem, IntTensor};

impl<G, F, I> IntTensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        BaseOps::<G>::empty(shape, device)
    }

    fn int_shape<const D: usize>(
        _tensor: &<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> Shape<D> {
        todo!()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Data<I, D> {
        BaseOps::<G>::to_data(&tensor)
    }

    fn int_from_data<const D: usize>(
        data: Data<I, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        BaseOps::<G>::from_data(data, device)
    }

    fn int_device<const D: usize>(
        _tensor: &<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::Device {
        todo!()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        BaseOps::<G>::to_device(tensor, device)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        _shape: Shape<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D2> {
        todo!()
    }

    fn int_index<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_index_assign<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
        _value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_mask_scatter<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _source: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mask_fill<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _value: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_gather<const D: usize>(
        _dim: usize,
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_scatter<const D: usize>(
        _dim: usize,
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_index_select_dim<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _dim: usize,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_index_select_dim_assign<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1>,
        _dim: usize,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
        _value: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D1> {
        todo!()
    }

    fn int_cat<const D: usize>(
        _tensors: Vec<<WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_greater_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_lower_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::IntElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        NumericOps::add::<I, D>(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        NumericOps::add_scalar(lhs, rhs)
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        NumericOps::sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        NumericOps::sub_scalar(lhs, rhs)
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        NumericOps::mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        NumericOps::mul_scalar(lhs, rhs)
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        NumericOps::div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        NumericOps::div_scalar(lhs, rhs)
    }

    fn int_neg<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_zeros<const D: usize>(
        _shape: Shape<D>,
        _device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_ones<const D: usize>(
        _shape: Shape<D>,
        _device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_sum<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_sum_dim<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_mean<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1> {
        todo!()
    }

    fn int_mean_dim<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmax<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn int_argmin<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
