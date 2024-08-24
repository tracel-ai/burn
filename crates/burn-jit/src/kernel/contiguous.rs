use crate::{tensor::JitTensor, JitElement, JitRuntime};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    if tensor.is_contiguous() {
        return tensor;
    }

    let output = cubecl::linalg::tensor::into_contiguous::<R, E::Primitive>(
        &tensor.client,
        tensor.as_handle_ref(),
    );

    JitTensor::new(
        tensor.client,
        output.handle,
        output.shape.into(),
        tensor.device,
        output.strides.try_into().unwrap(),
    )
}
