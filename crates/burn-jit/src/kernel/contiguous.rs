use crate::{tensor::JitTensor, JitElement, JitRuntime};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime, E: JitElement>(tensor: JitTensor<R, E>) -> JitTensor<R, E> {
    if tensor.is_contiguous() {
        return tensor;
    }

    let output =
        cubecl::linalg::tensor::into_contiguous::<R, E>(&tensor.client, tensor.as_handle_ref());

    JitTensor::new(
        tensor.client,
        output.handle,
        output.shape.into(),
        tensor.device,
        output.strides,
    )
}
