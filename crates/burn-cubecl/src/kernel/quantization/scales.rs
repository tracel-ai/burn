use burn_backend::{TensorMetadata, cubecl::dtype_to_storage_type, quantization::QuantScheme};
use burn_std::{QuantMode, params_shape};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

use crate::{
    kernel::{into_contiguous, utils::address_type},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// One block's symmetric scale per unit: the block's largest magnitude times `inverse_range`,
/// the value range's inverse, so the block's largest maps onto its format's largest value.
///
/// A block is `block` consecutive elements of the contiguous input, which is what one spanning
/// only the innermost axis is, and block `ABSOLUTE_POS`'s scale is element `ABSOLUTE_POS` of the
/// scales.
#[cube(launch_unchecked, address_type = "dynamic")]
fn symmetric_block_scales_kernel<F: Float, N: Size>(
    input: LinearView<'_, Vector<F, N>>,
    mut scales: LinearViewMut<'_, Vector<F, N>>,
    inverse_range: InputScalar,
    #[comptime] block: usize,
    #[define(F)] _dtype: ElemType,
) {
    if !scales.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let first = ABSOLUTE_POS * block;
    let mut most = Vector::<F, N>::new(F::new(0.0));
    #[unroll]
    for at in 0..block {
        most = max(most, input.read(first + at).abs());
    }
    scales.write(ABSOLUTE_POS, most * Vector::new(inverse_range.get::<F>()));
}

/// The block scales `quantize_dynamic` quantizes `tensor` under `scheme` with, in one launch: one
/// per block, its largest magnitude over the format's, as min-max calibration computes them.
///
/// `None` for a scheme it does not compute alone — a per-tensor level, which normalizes the
/// blocks against the largest of them; a block spanning more than the innermost axis; a mode other
/// than symmetric — whose scales the composed calibration computes instead.
pub fn symmetric_block_scales(tensor: &CubeTensor, scheme: &QuantScheme) -> Option<CubeTensor> {
    let block = scheme.block_size()?;
    let rank = tensor.meta.shape().num_dims();
    let dims = block.to_dim_vec(rank);
    let (inner, outer) = dims.split_last()?;
    let block = *inner as usize;
    let shape = tensor.shape();
    let innermost = outer.iter().all(|extent| *extent == 1)
        && block > 0
        && shape[rank - 1].is_multiple_of(block);
    if !innermost || scheme.tensor_scale().is_some() || !matches!(scheme.mode, QuantMode::Symmetric)
    {
        return None;
    }

    let input = into_contiguous(tensor.clone());
    let scales = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        params_shape(&shape, scheme),
        input.dtype,
    );
    let blocks = shape.num_elements() / block;
    let (low, high) = scheme.value.range();
    let inverse_range = InputScalar::new(2.0 / (high - low), dtype_to_storage_type(input.dtype));
    let cube_dim = CubeDim::new(&input.client, blocks);
    let cube_count = calculate_cube_count_elemwise(&input.client, blocks, cube_dim);
    let at = address_type!(input, scales);
    let dtype = dtype_to_storage_type(input.dtype);

    // SAFETY: each unit reads its block's `block` elements, which lie within the input since the
    // blocks tile it, and writes its one scale, bounded by the scales' extent.
    unsafe {
        symmetric_block_scales_kernel::launch_unchecked(
            &scales.client,
            cube_count,
            cube_dim,
            at,
            1,
            input.into_linear_view(),
            scales.clone().into_linear_view(),
            inverse_range,
            block,
            dtype,
        );
    }

    Some(scales)
}
