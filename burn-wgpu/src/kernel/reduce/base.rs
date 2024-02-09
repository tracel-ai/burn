use crate::{element::JitElement, tensor::JitTensor, Runtime};

/// Creates an empty output tensor with reduce output shape
pub fn init_reduce_output<R: Runtime, E: JitElement, const D: usize>(
    input: &JitTensor<R, E, D>,
    reduce_dim: usize,
) -> JitTensor<R, E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[reduce_dim] = 1;

    // Create output handle
    let num_elems_output = shape_out.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<E>());
    JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        handle,
    )
}
