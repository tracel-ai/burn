use crate::tensor::CubeTensor;
use crate::{CubeRuntime, FloatElement, ops::empty_qtensor};
use burn_tensor::quantization::QuantScheme;

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let output = empty_qtensor(tensor.shape.clone(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();

    cubecl_quant::quantize::launch_ref::<R, F>(
        &tensor.client,
        &tensor.as_handle_ref(),
        &out_values.as_handle_ref(),
        &scale.as_handle_ref(),
        &out_params.as_handle_ref(),
        scheme,
    );

    output
}
