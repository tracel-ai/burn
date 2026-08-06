use crate::CubeRuntime;
use crate::{kernel::cast, ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::{TensorMetadata, quantization::QuantScheme};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    global: Option<CubeTensor<R>>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let out_global = output.global();
    let dtype = tensor.dtype;

    // The kernel reads the incoming per-tensor scale at the precision the scheme stores one in,
    // unlike the block scales, which it reads at the tensor's float dtype.
    let global = match (global, out_global.as_ref()) {
        (Some(global), Some(out_global)) => Some(cast(global, out_global.dtype)),
        (global, _) => global,
    };

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        global.map(|g| g.binding()),
        out_params.binding(),
        out_global.map(|g| g.binding()),
        scheme,
        dtype_to_elem_type(dtype),
    )
    .expect("Kernel to never fail");

    output
}
