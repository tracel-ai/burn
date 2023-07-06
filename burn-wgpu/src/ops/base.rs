use crate::{
    element::WgpuElement,
    kernel::{self, cat, mask_fill, mask_fill_inplace, mask_where, mask_where_inplace},
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};
use burn_tensor::{backend::Backend, Data, Shape};
use std::marker::PhantomData;

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub struct Init<G: GraphicsApi> {
    _g: PhantomData<G>,
}

impl<G: GraphicsApi> Init<G> {
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
