use crate::{element::JitElement, kernel, tensor::JitTensor, Runtime};
use burn_tensor::{Data, Reader, Shape};

pub(crate) fn from_data<R: Runtime, E: JitElement, const D: usize>(
    data: Data<E, D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    let buffer = client.create(E::as_bytes(&data.value));

    JitTensor::new(client, device.clone(), data.shape, buffer)
}

pub(crate) fn into_data<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> Reader<Data<E, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor
        .client
        .read(&tensor.handle)
        .map(|bytes| Data::new(E::from_bytes(&bytes).to_vec(), tensor.shape))
}

pub(crate) fn bool_into_data<R: Runtime, const D: usize>(
    tensor: JitTensor<R, u32, D>,
) -> Reader<Data<bool, D>> {
    let tensor = kernel::into_contiguous(tensor);

    tensor.client.read(&tensor.handle).map(|bytes| {
        Data::new(
            u32::from_bytes(&bytes).iter().map(|i| *i != 0).collect(),
            tensor.shape,
        )
    })
}

pub(crate) fn to_device<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    if &tensor.device == device {
        return tensor;
    }

    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub(crate) fn empty<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new(client, device.clone(), shape, buffer)
}

pub(crate) fn swap_dims<R: Runtime, E: JitElement, const D: usize>(
    mut tensor: JitTensor<R, E, D>,
    dim1: usize,
    dim2: usize,
) -> JitTensor<R, E, D> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape.dims.swap(dim1, dim2);

    tensor
}

pub(crate) fn permute<R: Runtime, E: JitElement, const D: usize>(
    mut tensor: JitTensor<R, E, D>,
    axes: [usize; D],
) -> JitTensor<R, E, D> {
    // remap strides
    tensor.strides = axes.map(|i| tensor.strides[i]);

    // remap shape
    tensor.shape.dims = axes.map(|i| tensor.shape.dims[i]);

    tensor
}

pub(crate) fn reshape<R: Runtime, E: JitElement, const D1: usize, const D2: usize>(
    tensor: JitTensor<R, E, D1>,
    shape: Shape<D2>,
) -> JitTensor<R, E, D2> {
    // TODO: Not force standard layout all the time (improve performance).
    let tensor = kernel::into_contiguous(tensor);

    JitTensor::new(tensor.client, tensor.device, shape, tensor.handle)
}
