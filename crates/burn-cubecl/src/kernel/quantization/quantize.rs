use crate::CubeRuntime;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::{TensorMetadata, quantization::QuantScheme};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        out_params.binding(),
        scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
