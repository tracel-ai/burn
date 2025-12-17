// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_backend::tensor::BoolElem;
use burn_backend::{ElementConversion, TensorMetadata, tensor::FloatTensor};
use burn_backend::{
    backend::ExecutionError,
    ops::BoolTensorOps,
    tensor::{BoolTensor, IntTensor},
};
use ndarray::IntoDimension;

// Current crate
use crate::element::{FloatNdArrayElement, IntNdArrayElement, QuantElement};
use crate::{NdArray, execute_with_int_dtype, tensor::NdArrayTensor};
use crate::{NdArrayDevice, SharedArray};

// Workspace crates
use burn_backend::{Shape, TensorData, backend::Backend};

use super::{NdArrayBoolOps, NdArrayOps};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> BoolTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn bool_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor {
        if !data.dtype.is_bool() {
            unimplemented!("Unsupported dtype for `bool_from_data`")
        }
        NdArrayTensor::from_data(data)
    }

    async fn bool_into_data(tensor: NdArrayTensor) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_to_device(tensor: NdArrayTensor, _device: &NdArrayDevice) -> NdArrayTensor {
        tensor
    }

    fn bool_reshape(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        NdArrayOps::reshape(tensor.bool(), shape).into()
    }

    fn bool_slice(tensor: NdArrayTensor, slices: &[burn_backend::Slice]) -> NdArrayTensor {
        NdArrayOps::slice(tensor.bool(), slices).into()
    }

    fn bool_into_int(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use mapv directly instead of collecting to Vec and going through TensorData
        let int_array: SharedArray<I> = tensor.bool().mapv(|b| b.elem()).into_shared();
        int_array.into()
    }

    fn bool_device(_tensor: &NdArrayTensor) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn bool_empty(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor {
        Self::bool_zeros(shape, _device)
    }

    fn bool_zeros(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor {
        let values = vec![false; shape.num_elements()];
        NdArrayTensor::from_data(TensorData::new(values, shape))
    }

    fn bool_ones(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor {
        let values = vec![true; shape.num_elements()];
        NdArrayTensor::from_data(TensorData::new(values, shape))
    }

    fn bool_slice_assign(
        tensor: NdArrayTensor,
        slices: &[burn_backend::Slice],
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        NdArrayOps::slice_assign(tensor.bool(), slices, value.bool()).into()
    }

    fn bool_cat(tensors: Vec<NdArrayTensor>, dim: usize) -> NdArrayTensor {
        NdArrayOps::cat(tensors.into_iter().map(|it| it.bool()).collect(), dim).into()
    }

    fn bool_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        NdArrayBoolOps::equal(lhs.bool(), rhs.bool()).into()
    }

    fn bool_not(tensor: NdArrayTensor) -> NdArrayTensor {
        tensor.bool().mapv(|a| !a).into_shared().into()
    }

    fn bool_and(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        NdArrayBoolOps::and(lhs.bool(), rhs.bool()).into()
    }

    fn bool_or(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        NdArrayBoolOps::or(lhs.bool(), rhs.bool()).into()
    }

    fn bool_into_float(tensor: NdArrayTensor) -> FloatTensor<Self> {
        let arr: SharedArray<E> = tensor.bool().mapv(|b| b.elem()).into_shared();
        arr.into()
    }

    fn bool_swap_dims(tensor: NdArrayTensor, dim1: usize, dim2: usize) -> NdArrayTensor {
        NdArrayOps::swap_dims(tensor.bool(), dim1, dim2).into()
    }

    fn bool_permute(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        tensor.bool().permuted_axes(axes.into_dimension()).into()
    }

    fn bool_expand(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        NdArrayOps::expand(tensor.bool(), shape).into()
    }

    fn bool_select(tensor: NdArrayTensor, dim: usize, indices: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(indices, I, |indices: SharedArray<I>| -> NdArrayTensor {
            let tensor_bool = tensor.bool();
            let indices_vec: Vec<usize> = indices
                .into_iter()
                .map(|i| i.elem::<i64>() as usize)
                .collect();

            let selected = tensor_bool.select(ndarray::Axis(dim), &indices_vec);
            selected.into_shared().into()
        })
    }

    fn bool_select_or(
        tensor: NdArrayTensor,
        dim: usize,
        indices: NdArrayTensor,
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!(indices, I, |indices: SharedArray<I>| -> NdArrayTensor {
            let mut output_array = tensor.bool().into_owned();
            let value_bool = value.bool();

            for (index_value, index) in indices.into_iter().enumerate() {
                let index_usize = index.elem::<i64>() as usize;
                let mut view = output_array.index_axis_mut(ndarray::Axis(dim), index_usize);
                let value_slice = value_bool.index_axis(ndarray::Axis(dim), index_value);
                // For boolean tensors, select_assign should use logical OR operation
                view.zip_mut_with(&value_slice, |a, b| *a = *a || *b);
            }
            output_array.into_shared().into()
        })
    }

    fn bool_flip(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        NdArrayOps::flip(tensor.bool(), axes).into()
    }

    fn bool_unfold(tensor: NdArrayTensor, dim: usize, size: usize, step: usize) -> NdArrayTensor {
        NdArrayOps::unfold(tensor.bool(), dim, size, step).into()
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        NdArrayOps::mask_where(tensor.bool(), mask.bool(), value.bool()).into()
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolElem<Self>,
    ) -> BoolTensor<Self> {
        NdArrayOps::mask_fill(tensor.bool(), mask.bool(), value.elem()).into()
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        execute_with_int_dtype!(indices, |indices| NdArrayOps::gather(
            dim,
            tensor.bool(),
            indices
        ))
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        execute_with_int_dtype!(indices, |indices| NdArrayOps::scatter(
            dim,
            tensor.bool(),
            indices,
            value.bool()
        ))
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: BoolElem<Self>) -> BoolTensor<Self> {
        NdArrayBoolOps::equal_elem(lhs.bool(), rhs).into()
    }

    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        // Use view() for zero-copy on borrowed storage with short-circuit evaluation
        let result = NdArrayBoolOps::any_view(tensor.bool().view());
        NdArrayTensor::from_data(TensorData::new(vec![result], Shape::new([1])))
    }

    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        // Use view() for zero-copy on borrowed storage with short-circuit evaluation
        let result = NdArrayBoolOps::all_view(tensor.bool().view());
        NdArrayTensor::from_data(TensorData::new(vec![result], Shape::new([1])))
    }
}
