use crate::{
    comparison, comparison_elem, comparison_elem_inplace, comparison_inplace,
    element::WgpuElement,
    kernel::{
        self, cat, comparison, comparison_elem, comparison_elem_inplace, comparison_inplace,
        mask_fill, mask_fill_inplace, mask_where, mask_where_inplace,
    },
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};
use burn_tensor::{backend::Backend, Data, Shape};
use std::{marker::PhantomData, mem};

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub struct BaseOps<G: GraphicsApi> {
    _g: PhantomData<G>,
}

comparison!(Equal, "==");
comparison!(Greater, ">");
comparison!(GreaterEqual, ">=");
comparison!(Lower, "<");
comparison!(LowerEqual, "<=");

comparison_inplace!(EqualInplace, "==");
comparison_inplace!(GreaterInplace, ">");
comparison_inplace!(GreaterEqualInplace, ">=");
comparison_inplace!(LowerInplace, "<");
comparison_inplace!(LowerEqualInplace, "<=");

comparison_elem!(EqualElem, "==");
comparison_elem!(GreaterElem, ">");
comparison_elem!(GreaterEqualElem, ">=");
comparison_elem!(LowerElem, "<");
comparison_elem!(LowerEqualElem, "<=");

comparison_elem_inplace!(EqualElemInplace, "==");
comparison_elem_inplace!(GreaterElemInplace, ">");
comparison_elem_inplace!(GreaterEqualElemInplace, ">=");
comparison_elem_inplace!(LowerElemInplace, "<");
comparison_elem_inplace!(LowerEqualElemInplace, "<=");

impl<G: GraphicsApi> BaseOps<G> {
    pub fn from_data<E: WgpuElement, const D: usize>(
        data: Data<E, D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer_with_data_options(E::as_bytes(&data.value), true);

        WgpuTensor::new(context, data.shape, buffer)
    }

    pub fn into_data<E: WgpuElement, const D: usize>(tensor: WgpuTensor<E, D>) -> Data<E, D> {
        let tensor = kernel::into_continuous(tensor);
        let bytes = tensor.context.read_buffer(tensor.buffer);
        let values = E::from_bytes(&bytes);

        Data::new(values.to_vec(), tensor.shape)
    }

    pub fn to_device<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        if &tensor.context.device == device {
            return tensor;
        }

        let context = get_context::<G>(device);
        tensor.to_context(context)
    }

    pub fn empty<E: WgpuElement, const D: usize>(
        shape: Shape<D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

        WgpuTensor::new(context, shape, buffer)
    }

    pub fn swap_dims<E: WgpuElement, const D: usize>(
        mut tensor: WgpuTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> WgpuTensor<E, D> {
        tensor.strides.swap(dim1, dim2);

        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    pub fn reshape<E: WgpuElement, const D1: usize, const D2: usize>(
        tensor: WgpuTensor<E, D1>,
        shape: Shape<D2>,
    ) -> WgpuTensor<E, D2> {
        // TODO: Not force standard layout all the time (improve performance).
        let tensor = kernel::into_continuous(tensor);

        WgpuTensor::new(tensor.context, shape, tensor.buffer)
    }

    pub fn equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<EqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<EqualInplace, E, D>(rhs, lhs);
        }

        comparison::<Equal, E, D>(lhs, rhs)
    }

    pub fn greater<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<GreaterInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<LowerInplace, E, D>(rhs, lhs);
        }

        comparison::<Greater, E, D>(lhs, rhs)
    }

    pub fn greater_equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<GreaterEqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<LowerEqualInplace, E, D>(rhs, lhs);
        }

        comparison::<GreaterEqual, E, D>(lhs, rhs)
    }

    pub fn lower<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<LowerInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<GreaterInplace, E, D>(rhs, lhs);
        }

        comparison::<Lower, E, D>(lhs, rhs)
    }

    pub fn lower_equal<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<u32, D> {
        let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

        if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
            return comparison_inplace::<LowerEqualInplace, E, D>(lhs, rhs);
        }
        if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
            return comparison_inplace::<GreaterEqualInplace, E, D>(rhs, lhs);
        }

        comparison::<LowerEqual, E, D>(lhs, rhs)
    }

    pub fn equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<EqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<EqualElem, E, D>(lhs, rhs)
    }

    pub fn greater_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<GreaterElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<GreaterElem, E, D>(lhs, rhs)
    }

    pub fn lower_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<LowerElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<LowerElem, E, D>(lhs, rhs)
    }

    pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<GreaterEqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<GreaterEqualElem, E, D>(lhs, rhs)
    }

    pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<u32, D> {
        if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
            return comparison_elem_inplace::<LowerEqualElemInplace, E, D>(lhs, rhs);
        }

        comparison_elem::<LowerEqualElem, E, D>(lhs, rhs)
    }

    pub fn mask_fill<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        mask: WgpuTensor<u32, D>,
        value: E,
    ) -> WgpuTensor<E, D> {
        if tensor.can_mut() {
            return mask_fill_inplace(tensor, mask, value);
        }

        mask_fill(tensor, mask, value)
    }

    pub fn mask_where<E: WgpuElement, const D: usize>(
        tensor: WgpuTensor<E, D>,
        mask: WgpuTensor<u32, D>,
        value: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        if tensor.can_mut_broadcast(&value) {
            return mask_where_inplace(tensor, mask, value, false);
        }
        if value.can_mut_broadcast(&tensor) {
            return mask_where_inplace(value, mask, tensor, true);
        }

        mask_where(tensor, mask, value)
    }

    pub fn cat<E: WgpuElement, const D: usize>(
        tensors: Vec<WgpuTensor<E, D>>,
        dim: usize,
    ) -> WgpuTensor<E, D> {
        cat(tensors, dim)
    }
}
