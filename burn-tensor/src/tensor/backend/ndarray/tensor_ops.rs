use super::{BatchMatrix, NdArrayBackend, NdArrayTensor};
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
        let array = &lhs.array + &rhs.array;
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }

    fn add_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array + *rhs;
        let array = array.into_shared();
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }

    fn sub<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array - &rhs.array;
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }

    fn sub_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array - *rhs;
        let array = array.into_shared();
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }

    fn mul<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array * &rhs.array;
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }

    fn mul_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array * *rhs;
        let array = array.into_shared();
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }

    fn div<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array / &rhs.array;
        let array = array.into_shared();
        let shape = lhs.shape.higher(&rhs.shape);

        NdArrayTensor { array, shape }
    }

    fn div_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array / *rhs;
        let array = array.into_shared();
        let shape = lhs.shape;

        NdArrayTensor { array, shape }
    }

    fn matmul<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let batch_self = BatchMatrix::from_ndarray(lhs.array.clone(), lhs.shape);
        let batch_other = BatchMatrix::from_ndarray(rhs.array.clone(), rhs.shape);

        let self_iter = batch_self.arrays.iter();
        let other_iter = batch_other.arrays.iter();
        let arrays = self_iter
            .zip(other_iter)
            .map(|(lhs, rhs)| lhs.dot(rhs))
            .map(|output| output.into_shared())
            .collect();

        let mut shape = lhs.shape;
        shape.dims[D - 1] = rhs.shape.dims[D - 1];
        let output = BatchMatrix::new(arrays, shape);

        NdArrayTensor::from_bmatrix(output)
    }
}
