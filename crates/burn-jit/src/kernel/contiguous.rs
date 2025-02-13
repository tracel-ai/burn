use crate::{execute_with_dtype, tensor::CubeTensor, CubeRuntime};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if tensor.is_contiguous() {
        return tensor;
    }

    execute_with_dtype!(tensor.dtype, E, {
        let output = cubecl::linalg::tensor::into_contiguous::<R, E>(
            &tensor.client,
            &tensor.as_handle_ref(),
        );

        CubeTensor::new(
            tensor.client,
            output.handle,
            output.shape.into(),
            tensor.device,
            output.strides,
            tensor.dtype,
        )
    })
}
