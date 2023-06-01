use burn_tensor::{backend::Backend, ops::BoolTensorOps, Data, Shape};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend,
};

use super::{BaseOps, BoolTensor, Device, IntTensor};

impl<G, F, I> BoolTensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        BaseOps::<G>::empty(shape, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<Self, D>) -> Data<bool, D> {
        let data = BaseOps::<G>::to_data(&tensor);

        Data::new(data.value.into_iter().map(|i| i != 0).collect(), data.shape)
    }

    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let data: Data<u32, D> = Data::new(
            data.value
                .into_iter()
                .map(|c| match c {
                    true => 1,
                    false => 0,
                })
                .collect(),
            data.shape,
        );
        BaseOps::<G>::from_data(data, device)
    }

    fn bool_into_int<const D: usize>(_tensor: BoolTensor<Self, D>) -> IntTensor<Self, D> {
        todo!()
    }

    fn bool_device<const D: usize>(
        _tensor: &<WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::Device {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D1>,
        _shape: Shape<D2>,
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
