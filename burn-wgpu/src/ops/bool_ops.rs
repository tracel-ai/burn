use burn_tensor::{backend::Backend, ops::BoolTensorOps};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend,
};

impl<G, F, I> BoolTensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn bool_empty<const D: usize>(
        _shape: burn_tensor::Shape<D>,
        _device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_shape<const D: usize>(
        _tensor: &<WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> burn_tensor::Data<bool, D> {
        todo!()
    }

    fn bool_from_data<const D: usize>(
        _data: burn_tensor::Data<bool, D>,
        _device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_into_int<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn bool_device<const D: usize>(
        _tensor: &<WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::Device {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _device: &<WGPUBackend<G, F, I> as Backend>::Device,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1>,
        _shape: burn_tensor::Shape<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D2> {
        todo!()
    }

    fn bool_index<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_index_assign<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
        _value: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        _tensors: Vec<<WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _rhs: bool,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }
}
