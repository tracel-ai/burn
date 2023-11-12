use super::padding::{crop, pad_round, PaddingOutput};
use crate::{
    compute::{DynamicKernel, WgpuHandle, WorkGroup},
    element::WgpuElement,
    kernel::{build_info, into_contiguous, matmul::utils::shape_out, DynamicKernelSource},
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::{Element, Shape};

pub(crate) const B_M: usize = 64;
pub(crate) const B_N: usize = 64;
pub(crate) const B_K: usize = 32;
pub(crate) const WORKGROUP_SIZE: usize = 16;

pub(super) fn make_workgroup<const D: usize>(output_shape: &Shape<D>) -> WorkGroup {
    let num_blocks_x = f32::ceil(output_shape.dims[D - 2] as f32 / B_M as f32) as u32;
    let num_blocks_y = f32::ceil(output_shape.dims[D - 1] as f32 / B_N as f32) as u32;
    let mut num_blocks_z = 1;
    for i in 0..D - 2 {
        num_blocks_z *= output_shape.dims[i];
    }

    WorkGroup::new(num_blocks_x, num_blocks_y, num_blocks_z as u32)
}

pub(super) fn make_info_handle<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
    output: &WgpuTensor<E, D>,
) -> WgpuHandle {
    let info = build_info(&[lhs, rhs, output]);
    rhs.client.create(bytemuck::cast_slice(&info))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_tiling_2d_launch<
    E: WgpuElement + Element,
    const D: usize,
    K: DynamicKernelSource + 'static,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    output: WgpuTensor<E, D>,
    kernel: K,
) -> WgpuTensor<E, D> {
    // A tensor may need to be padded, in which case it will implicitly become contiguous
    // If not needed, it is only turned into contiguous if some batch dim has been swapped with row or col dim.
    // If batches were swapped among themselves, or if the last two dims are transposed, the underlying
    // kernel handles it without needing to turn it into contiguous.
    let round_lhs = pad_round(lhs, B_M, B_K);
    let lhs = match round_lhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_lhs.into_tensor(),
    };
    let round_rhs = pad_round(rhs, B_K, B_N);
    let rhs = match round_rhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_rhs.into_tensor(),
    };

    let rounded_output_shape = shape_out(&lhs, &rhs);

    let rounded_output = empty_device(
        rhs.client.clone(),
        rhs.device.clone(),
        rounded_output_shape.clone(),
    );

    let workgroup = make_workgroup(&rounded_output_shape);
    let info_handle = make_info_handle(&lhs, &rhs, &rounded_output);

    lhs.client.execute(
        Box::new(DynamicKernel::new(kernel, workgroup)),
        &[
            &lhs.handle,
            &rhs.handle,
            &rounded_output.handle,
            &info_handle,
        ],
    );

    crop(rounded_output, output)
}
