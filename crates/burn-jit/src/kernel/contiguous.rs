use crate::{kernel_with_dtype, tensor::JitTensor, JitRuntime};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime>(tensor: JitTensor<R>) -> JitTensor<R> {
    if tensor.is_contiguous() {
        return tensor;
    }

    kernel_with_dtype!(tensor.dtype, |E| {
        let output =
            cubecl::linalg::tensor::into_contiguous::<R, E>(&tensor.client, tensor.as_handle_ref());

        JitTensor::new(
            tensor.client,
            output.handle,
            output.shape.into(),
            tensor.device,
            output.strides,
            tensor.dtype,
        )
    })
}
