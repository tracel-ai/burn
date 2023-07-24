use crate::{
    element::WgpuElement, kernel, pool::get_context, tensor::WgpuTensor, GraphicsApi, WgpuDevice,
};
use burn_tensor::{backend::Backend, Data, Shape};

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type FullPrecisionBackend<B> = <B as Backend>::FullPrecisionBackend;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;
pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub fn from_data<G: GraphicsApi, E: WgpuElement, const D: usize>(
    data: Data<E, D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    let context = get_context::<G>(device);
    let buffer = context.create_buffer_with_data_options(E::as_bytes(&data.value), true);

    WgpuTensor::new(context, data.shape, buffer)
}

pub fn into_data<E: WgpuElement, const D: usize>(tensor: WgpuTensor<E, D>) -> Data<E, D> {
    let tensor = kernel::into_contiguous(tensor);
    let bytes = tensor.context.read_buffer(tensor.buffer);
    let values = E::from_bytes(&bytes);

    Data::new(values.to_vec(), tensor.shape)
}

pub fn to_device<G: GraphicsApi, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    if &tensor.context.device == device {
        return tensor;
    }

    let context = get_context::<G>(device);
    tensor.to_context(context)
}

pub fn empty<G: GraphicsApi, E: WgpuElement, const D: usize>(
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
    let tensor = kernel::into_contiguous(tensor);

    WgpuTensor::new(tensor.context, shape, tensor.buffer)
}
