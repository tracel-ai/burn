use crate::{
    ops::{empty_qtensor_optimized, reshape},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::{
    TensorMetadata,
    quantization::{QuantScheme, params_shape},
};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
///
/// `scale` may be flat: the kernel reads it on the scheme's grid of blocks over `tensor`.
///
/// # Panics
///
/// Where `scale` holds another number of scales than that grid has blocks.
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
    let scale = on_grid(scale, &tensor, scheme);

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

/// `scale` stated on the grid of blocks `scheme` lays over `tensor`, which is what the kernel
/// indexes it by: a flat scale of a block scheme would be read along the wrong axes.
fn on_grid(scale: CubeTensor, tensor: &CubeTensor, scheme: &QuantScheme) -> CubeTensor {
    let grid = params_shape(&tensor.shape(), scheme);
    assert_eq!(
        scale.shape().num_elements(),
        grid.num_elements(),
        "quantize: {scheme:?} over {:?} takes one scale for each of {grid:?} blocks",
        tensor.shape()
    );
    match scale.shape() == grid {
        true => scale,
        false => reshape(scale, grid),
    }
}
