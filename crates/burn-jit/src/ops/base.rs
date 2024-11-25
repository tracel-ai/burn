use crate::{element::BasicJitElement, kernel, tensor::JitTensor, JitRuntime};
use burn_tensor::{ElementConversion, Shape, TensorData};
use cubecl::{
    calculate_cube_count_elemwise,
    client::ComputeClient,
    cube,
    prelude::{Numeric, ScalarArg, Tensor, Vectorized, ABSOLUTE_POS},
    tensor_vectorization_factor, CubeDim,
};

pub(crate) fn from_data<R: JitRuntime, E: BasicJitElement>(
    data: TensorData,
    device: &R::Device,
) -> JitTensor<R> {
    let shape: Shape = (&data.shape).into();
    let client = R::client(device);
    let elems: Vec<E> = data.convert::<E>().into_vec().unwrap();
    let buffer = client.create(&E::to_elem_data(&elems));
    JitTensor::new_contiguous(client, device.clone(), shape, buffer, E::dtype())
}

pub(crate) async fn into_data<R: JitRuntime, E: BasicJitElement>(
    tensor: JitTensor<R>,
) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);
    let elements = E::from_elem_data(tensor.client.read_one_async(tensor.handle.binding()).await);
    TensorData::new(elements, tensor.shape)
}

pub(crate) fn to_device<R: JitRuntime>(tensor: JitTensor<R>, device: &R::Device) -> JitTensor<R> {
    if &tensor.device == device {
        return tensor;
    }

    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub(crate) fn swap_dims<R: JitRuntime>(
    mut tensor: JitTensor<R>,
    dim1: usize,
    dim2: usize,
) -> JitTensor<R> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape.dims.swap(dim1, dim2);

    tensor
}

pub(crate) fn permute<R: JitRuntime>(mut tensor: JitTensor<R>, axes: &[usize]) -> JitTensor<R> {
    // remap strides
    tensor.strides = axes.iter().map(|i| tensor.strides[*i]).collect();

    // remap shape
    tensor.shape.dims = axes.iter().map(|i| tensor.shape.dims[*i]).collect();

    tensor
}
pub(crate) fn expand<R: JitRuntime>(tensor: JitTensor<R>, target_shape: Shape) -> JitTensor<R> {
    let ndims_in = tensor.shape.num_dims();
    let ndims_out = target_shape.num_dims();

    // Initialize new strides with zeros
    let mut new_strides = vec![0usize; ndims_out];

    // Calculate the difference in dimensions
    let dim_diff = ndims_out.saturating_sub(ndims_in);

    // Compare dimensions from the end, setting strides for matching dimensions or broadcasted ones
    let mut tensor_dim_iter = tensor.shape.dims.iter().rev();
    for i in (0..ndims_out).rev() {
        if i >= dim_diff {
            if let Some(&tensor_dim) = tensor_dim_iter.next() {
                if tensor_dim == target_shape.dims[i] || tensor_dim == 1 {
                    // Copy stride for non-broadcast dimensions or set to 0 for broadcast ones
                    new_strides[i] = if tensor_dim == target_shape.dims[i] {
                        tensor.strides[i - dim_diff]
                    } else {
                        0
                    };
                } else {
                    // Error handling: Dimension mismatch for broadcasting
                    panic!(
                        "Dimension mismatch: cannot broadcast dimension {} of tensor to target shape",
                        tensor_dim
                    );
                }
            } else {
                // If the input tensor has fewer dimensions, treat missing dimensions as 1
                // and set stride to 0 (broadcasting)
                new_strides[i] = 0;
            }
        } else {
            // For extra dimensions in the target shape, set stride to 0 (broadcasting)
            new_strides[i] = 0;
        }
    }

    JitTensor {
        client: tensor.client,
        device: tensor.device,
        shape: target_shape,
        strides: new_strides,
        handle: tensor.handle,
        dtype: tensor.dtype,
    }
}

pub(crate) fn reshape<R: JitRuntime>(tensor: JitTensor<R>, shape: Shape) -> JitTensor<R> {
    // TODO: Not force standard layout all the time (improve performance).
    let tensor = kernel::into_contiguous(tensor);

    JitTensor::new_contiguous(
        tensor.client,
        tensor.device,
        shape,
        tensor.handle,
        tensor.dtype,
    )
}

pub fn full<R: JitRuntime, E: BasicJitElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> JitTensor<R> {
    let client = R::client(device);
    full_device::<R, E>(client, shape, device.clone(), value)
}

pub fn full_device<R: JitRuntime, E: BasicJitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> JitTensor<R> {
    let ndims = shape.num_dims();
    let empty = empty_device::<R, E>(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric + Vectorized>(tensor: &mut Tensor<C>, value: C) {
        if ABSOLUTE_POS >= tensor.len() {
            return;
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &empty.shape.dims, &empty.strides, ndims - 1);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);

    full_kernel::launch::<E, R>(
        &empty.client,
        cube_count,
        cube_dim,
        empty.as_tensor_arg::<E>(vectorization_factor),
        ScalarArg::new(value),
    );

    empty
}

pub fn zeros<R: JitRuntime, E: BasicJitElement>(shape: Shape, device: &R::Device) -> JitTensor<R> {
    let client = R::client(device);
    zeros_device::<R, E>(client, device.clone(), shape)
}

pub fn zeros_device<R: JitRuntime, E: BasicJitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    full_device::<R, E>(client, shape, device, 0.elem())
}

pub fn ones<R: JitRuntime, E: BasicJitElement>(shape: Shape, device: &R::Device) -> JitTensor<R> {
    let client = R::client(device);

    ones_device::<R, E>(client, device.clone(), shape)
}

pub fn ones_device<R: JitRuntime, E: BasicJitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    full_device::<R, E>(client, shape, device, 1.elem())
}

pub fn empty<R: JitRuntime, E: BasicJitElement>(shape: Shape, device: &R::Device) -> JitTensor<R> {
    let client = R::client(device);
    empty_device::<R, E>(client, device.clone(), shape)
}

pub fn empty_device<R: JitRuntime, E: BasicJitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape,
) -> JitTensor<R> {
    let buffer = client.empty(shape.num_elements() * E::as_elem().size());
    JitTensor::new_contiguous(client, device, shape, buffer, E::dtype())
}
