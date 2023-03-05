use alloc::vec::Vec;
use core::{marker::PhantomData, ops::Range};

use burn_tensor::Shape;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::SliceInfoElem;

use crate::{tensor::NdArrayTensor, to_nd_array_tensor};

pub struct NdArrayOps<E> {
    e: PhantomData<E>,
}

impl<E> NdArrayOps<E>
where
    E: Copy,
{
    pub fn index<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(indexes);
        let array = tensor.array.slice_move(slices.as_slice()).into_shared();

        NdArrayTensor { array }
    }

    pub fn index_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        let slices = Self::to_slice_args::<D1, D2>(indexes);
        let mut array = tensor.array.to_owned();
        array.slice_mut(slices.as_slice()).assign(&value.array);
        let array = array.into_owned().into_shared();

        NdArrayTensor { array }
    }

    pub fn reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
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

    pub fn cat<const D: usize>(
        tensors: Vec<NdArrayTensor<E, D>>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        let arrays: Vec<ndarray::ArrayView<E, IxDyn>> =
            tensors.iter().map(|t| t.array.view()).collect();
        let array = ndarray::concatenate(Axis(dim), &arrays)
            .unwrap()
            .into_shared();

        NdArrayTensor { array }
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
}
