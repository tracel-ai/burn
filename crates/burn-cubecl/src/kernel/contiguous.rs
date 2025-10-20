use burn_tensor::quantization::QTensorPrimitive;

use crate::{CubeRuntime, execute_with_dtype, ops::empty_qtensor, tensor::CubeTensor};

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if tensor.is_contiguous() {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor);
    }

    execute_with_dtype!(tensor.dtype, E, {
        let output =
            cubecl::std::tensor::into_contiguous::<R, E>(&tensor.client, &tensor.as_handle_ref());

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

/// Make a jit tensor contiguous with an aligned last stride. Tensor is considered already contiguous
/// if runtime can read it as is. This is equivalent in practice.
pub fn into_contiguous_aligned<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    if R::can_read_tensor(&tensor.shape, &tensor.strides) {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor);
    }

    execute_with_dtype!(tensor.dtype, E, {
        let output = cubecl::std::tensor::into_contiguous_pitched::<R, E>(
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

fn into_contiguous_quantized<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    execute_with_dtype!(tensor.dtype, E, {
        let output = empty_qtensor(tensor.shape.clone(), *tensor.scheme(), &tensor.device);

        cubecl::std::tensor::into_contiguous_ref::<R, E>(
            &tensor.client,
            &tensor.as_handle_ref(),
            &output.as_handle_ref(),
        );
        let scales = tensor.scales().unwrap();
        let out_scales = output.scales().unwrap();
        cubecl::std::tensor::into_contiguous_ref::<R, E>(
            &scales.client,
            &scales.as_handle_ref(),
            &out_scales.as_handle_ref(),
        );

        output
    })
}
