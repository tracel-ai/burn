use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::{TensorMetadata, quantization::QuantScheme};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize(
    tensor: CubeTensor,
    scheme: &QuantScheme,
    scale: CubeTensor,
    global: Option<CubeTensor>,
) -> CubeTensor {
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let out_global = output.global();
    let dtype = tensor.dtype;

    // Innermost first: the block scales, then the per-tensor scale they are normalized against.
    let mut scales = vec![scale.binding()];
    scales.extend(global.map(|global| global.binding()));

    let mut out_scales = vec![out_params.binding()];
    out_scales.extend(out_global.map(|global| global.binding()));

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        &scales,
        &out_scales,
        scheme,
        dtype_to_elem_type(dtype),
    )
    .expect("Kernel to never fail");

    output
}
