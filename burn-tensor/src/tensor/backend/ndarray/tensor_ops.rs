use std::ops::Range;

use super::{BatchMatrix, NdArrayBackend, NdArrayTensor};
use crate::{
    backend::{Backend, NdArrayDevice},
    ops::TensorOps,
    to_nd_array_tensor, Data, ElementConversion, NdArrayElement, Shape,
};
use ndarray::{Dim, SliceInfoElem};

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

    fn neg<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        Self::mul_scalar(tensor, &(-1f32).to_elem::<E>())
    }
    fn swap_dims<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<E, D> {
        let mut shape = tensor.shape;
        let dim1_new = shape.dims[dim2];
        let dim2_new = shape.dims[dim1];

        shape.dims[dim1] = dim1_new;
        shape.dims[dim2] = dim2_new;

        let mut array = tensor.array.clone();
        array.swap_axes(dim1, dim2);

        NdArrayTensor { array, shape }
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<E, D2> {
        match D2 {
            1 => to_nd_array_tensor!(1, shape, tensor.array),
            2 => to_nd_array_tensor!(2, shape, tensor.array),
            3 => to_nd_array_tensor!(3, shape, tensor.array),
            4 => to_nd_array_tensor!(4, shape, tensor.array),
            5 => to_nd_array_tensor!(5, shape, tensor.array),
            6 => to_nd_array_tensor!(6, shape, tensor.array),
            _ => panic!("NdArrayTensor support only 6 dimensions."),
        }
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        let shape = tensor.shape.index(indexes.clone());
        let slices = to_slice_args::<D1, D2>(indexes);
        let array = tensor
            .array
            .clone()
            .slice_move(slices.as_slice())
            .into_shared();

        NdArrayTensor { array, shape }
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: &NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        let slices = to_slice_args::<D1, D2>(indexes);
        let mut array = tensor.array.to_owned();
        array.slice_mut(slices.as_slice()).assign(&value.array);
        let array = array.into_owned().into_shared();

        let shape = tensor.shape;

        NdArrayTensor { array, shape }
    }

    fn mask_fill<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        mask: &NdArrayTensor<bool, D>,
        value: E,
    ) -> NdArrayTensor<E, D> {
        let elem = E::default();
        let mask_mul = mask.array.mapv(|x| match x {
            true => E::zeros(&elem),
            false => E::ones(&elem),
        });
        let mask_add = mask.array.mapv(|x| match x {
            true => value,
            false => E::zeros(&elem),
        });
        let array = (tensor.array.clone() * mask_mul) + mask_add;

        NdArrayTensor {
            array,
            shape: tensor.shape,
        }
    }

    fn equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());

        Self::equal_scalar(&tensor, &zero)
    }

    fn equal_scalar<const D: usize>(lhs: &NdArrayTensor<E, D>, rhs: &E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a == *rhs).into_shared();

        NdArrayTensor {
            shape: lhs.shape,
            array,
        }
    }

    fn greater<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::greater_scalar(&tensor, &zero)
    }

    fn greater_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a > *rhs).into_shared();

        NdArrayTensor {
            shape: lhs.shape,
            array,
        }
    }

    fn greater_equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::greater_equal_scalar(&tensor, &zero)
    }

    fn greater_equal_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a >= *rhs).into_shared();

        NdArrayTensor {
            shape: lhs.shape,
            array,
        }
    }

    fn lower<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::lower_scalar(&tensor, &zero)
    }

    fn lower_scalar<const D: usize>(lhs: &NdArrayTensor<E, D>, rhs: &E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a < *rhs).into_shared();

        NdArrayTensor {
            shape: lhs.shape,
            array,
        }
    }

    fn lower_equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::lower_equal_scalar(&tensor, &zero)
    }

    fn lower_equal_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a <= *rhs).into_shared();

        NdArrayTensor {
            shape: lhs.shape,
            array,
        }
    }
}

fn to_slice_args<const D1: usize, const D2: usize>(
    indexes: [Range<usize>; D2],
) -> [SliceInfoElem; D1] {
    let mut slices = [SliceInfoElem::NewAxis; D1];
    for i in 0..D1 {
        if i >= D2 {
            slices[i] = SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            }
        } else {
            slices[i] = SliceInfoElem::Slice {
                start: indexes[i].start as isize,
                end: Some(indexes[i].end as isize),
                step: 1,
            }
        }
    }
    slices
}
