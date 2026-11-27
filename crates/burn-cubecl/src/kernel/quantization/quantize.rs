use crate::{
    ops::{empty_qtensor_optimized, reshape},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::{
    TensorMetadata,
    quantization::{QuantScheme, level_shape},
};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
///
/// `scale` holds one scale for each block of the scheme's finest grid over `tensor`, and
/// `coarser` one for each block of its coarser grid, each in any shape that lists them in
/// row-major grid order, a flat one included.
///
/// # Panics
///
/// Where `coarser` is given for a one-level scheme or missing for a two-level one, and where a
/// level's scales hold another number of scales than its grid has blocks.
pub fn quantize(
    tensor: CubeTensor,
    scheme: &QuantScheme,
    scale: CubeTensor,
    coarser: Option<CubeTensor>,
) -> CubeTensor {
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let out_coarser = output.coarser();
    let dtype = tensor.dtype;

    let given: Vec<CubeTensor> = core::iter::once(scale).chain(coarser).collect();
    assert_eq!(
        given.len(),
        scheme.levels().count(),
        "quantize: {scheme:?} takes scales for each of its levels, got {}",
        given.len()
    );
    // The kernel finds a value's scale along the scales' own axes, so it reads each level's on
    // its grid. Finest first, the order the kernel counts.
    let scales: Vec<_> = (scheme.levels().zip(given))
        .map(|(level, scales)| {
            let grid = level_shape(&tensor.shape(), &level);
            assert_eq!(
                scales.shape().num_elements(),
                grid.num_elements(),
                "quantize: {scheme:?} over {:?} takes one scale for each of {grid:?} blocks",
                tensor.shape()
            );
            reshape(scales, grid).binding()
        })
        .collect();

    let mut out_scales = vec![out_params.binding()];
    out_scales.extend(out_coarser.map(|coarser| coarser.binding()));

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
