use crate::{element::WgpuElement, kernel, tensor::WgpuTensor, JitRuntime};
use burn_tensor::{Data, Reader, Shape};

pub fn from_data<B: JitRuntime, E: WgpuElement, const D: usize>(
    data: Data<E, D>,
    device: &B::Device,
) -> WgpuTensor<B, E, D> {
    let client = B::client(device);
    let buffer = client.create(E::as_bytes(&data.value));

    WgpuTensor::new(client, device.clone(), data.shape, buffer)
}

pub fn into_data<B: JitRuntime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<B, E, D>,
) -> Reader<Data<E, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor
        .client
        .read(&tensor.handle)
        .map(|bytes| Data::new(E::from_bytes(&bytes).to_vec(), tensor.shape))
}

pub fn bool_into_data<B: JitRuntime, const D: usize>(
    tensor: WgpuTensor<B, u32, D>,
) -> Reader<Data<bool, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor.client.read(&tensor.handle).map(|bytes| {
        Data::new(
            u32::from_bytes(&bytes).iter().map(|i| *i != 0).collect(),
            tensor.shape,
        )
    })
}

pub fn to_device<B: JitRuntime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<B, E, D>,
    device: &B::Device,
) -> WgpuTensor<B, E, D> {
    if &tensor.device == device {
        return tensor;
    }

    let client = B::client(device);
    tensor.to_client(client, device.clone())
}

pub fn empty<B: JitRuntime, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &B::Device,
) -> WgpuTensor<B, E, D> {
    let client = B::client(device);
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(client, device.clone(), shape, buffer)
}

pub fn swap_dims<B: JitRuntime, E: WgpuElement, const D: usize>(
    mut tensor: WgpuTensor<B, E, D>,
    dim1: usize,
    dim2: usize,
) -> WgpuTensor<B, E, D> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape.dims.swap(dim1, dim2);

    tensor
}

pub fn reshape<B: JitRuntime, E: WgpuElement, const D1: usize, const D2: usize>(
    tensor: WgpuTensor<B, E, D1>,
    shape: Shape<D2>,
) -> WgpuTensor<B, E, D2> {
    // TODO: Not force standard layout all the time (improve performance).
    let tensor = kernel::into_contiguous(tensor);

    WgpuTensor::new(tensor.client, tensor.device, shape, tensor.handle)
}
