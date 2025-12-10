use crate::CubeRuntime;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_tensor::quantization::QuantScheme;

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape.clone(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();

    cubek::quantization::quantize::launch_ref(
        &tensor.client,
        &tensor.as_handle_ref(),
        &out_values.as_handle_ref(),
        &scale.as_handle_ref(),
        &out_params.as_handle_ref(),
        scheme,
        tensor.dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
