use crate::{
    compute::compute_client, element::WgpuElement, kernel, tensor::WgpuTensor, GraphicsApi,
    WgpuDevice,
};
use burn_tensor::{backend::Backend, Data, Reader, Shape};

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
    let client = compute_client::<G>(device);
    let buffer = client.create(E::as_bytes(&data.value));

    WgpuTensor::new(client, device.clone(), data.shape, buffer)
}

pub fn into_data<E: WgpuElement, const D: usize>(tensor: WgpuTensor<E, D>) -> Reader<Data<E, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor
        .client
        .read(&tensor.handle)
        .map(|bytes| Data::new(E::from_bytes(&bytes).to_vec(), tensor.shape))
}

pub fn bool_into_data<const D: usize>(tensor: WgpuTensor<u32, D>) -> Reader<Data<bool, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor.client.read(&tensor.handle).map(|bytes| {
        Data::new(
            u32::from_bytes(&bytes).iter().map(|i| *i != 0).collect(),
            tensor.shape,
        )
    })
}

pub fn to_device<G: GraphicsApi, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    if &tensor.device == device {
        return tensor;
    }

    let client = compute_client::<G>(device);
    tensor.to_client(client, device.clone())
}

pub fn empty<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    let client = compute_client::<G>(device);
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(client, device.clone(), shape, buffer)
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

    WgpuTensor::new(tensor.client, tensor.device, shape, tensor.handle)
}
