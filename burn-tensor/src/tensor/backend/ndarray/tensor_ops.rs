use super::{NdArrayBackend, NdArrayTensor};
use crate::{
    backend::{Backend, NdArrayDevice},
    ops::TensorOps,
    Data, NdArrayElement, Shape,
};

impl<E: NdArrayElement> TensorOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn shape<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> &Shape<D> {
        &tensor.shape
    }

    fn to_data<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::Elem, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape)
    }

    fn into_data<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::Elem, D> {
        let values = tensor.array.into_iter().collect();
        Data::new(values, tensor.shape)
    }

    fn bool_shape<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> &Shape<D> {
        &tensor.shape
    }

    fn bool_to_data<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape)
    }

    fn bool_into_data<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values = tensor.array.into_iter().collect();
        Data::new(values, tensor.shape)
    }
    fn device<const D: usize>(_tensor: &NdArrayTensor<E, D>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn to_device<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        _device: NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        tensor.clone()
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: <NdArrayBackend<E> as Backend>::Device,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        NdArrayBackend::<E>::zeros(shape, device)
    }

    fn add<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = lhs.array.clone() + rhs.array.clone();
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }
    fn add_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = lhs.array.clone() + *rhs;
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }
    fn sub<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = lhs.array.clone() - rhs.array.clone();
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }
    fn sub_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = lhs.array.clone() - *rhs;
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }
}
