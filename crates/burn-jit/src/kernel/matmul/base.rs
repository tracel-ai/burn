use burn_compute::server::Handle;
use burn_tensor::Shape;

use crate::{
    compute::{DynamicKernel, WorkGroup},
    kernel::{build_info, into_contiguous, DynamicKernelSource},
    ops::numeric::empty_device,
    tensor::JitTensor,
    JitElement, Runtime,
};

use super::{
    init_matmul_output, matmul_autotune, matmul_simple, matmul_tiling_2d,
    padding::{crop, pad_round, PaddingOutput},
    shape_out,
    tiling2d_padded::matmul_tiling_2d_padded,
};

/// The strategy to be used when launching a matmul kernel.
#[derive(Default)]
pub enum MatmulStrategy {
    /// A simple kernel will be used with memory coalescing optimization.
    Simple {
        /// Number of invocations in x
        grid_x: usize,
        /// Number of invocations in y
        grid_y: usize,
    },
    /// A tiling 2d kernel will be used, with support for any matrix size without padding.
    Tiling2d {
        /// Number of invocations in x
        grid_x: usize,
        /// Number of invocations in y
        grid_y: usize,
        /// Block size along dimension of lhs
        block_size_m: usize,
        /// Block size along common dimension
        block_size_k: usize,
        /// Block size along dimension of rhs
        block_size_n: usize,
    },
    /// A tiling 2d kernel will be used, with support for any matrix size with padding.
    Tiling2dPadded {
        /// Number of invocations in x
        grid_x: usize,
        /// Number of invocations in y
        grid_y: usize,
        /// Block size along dimension of lhs
        block_size_m: usize,
        /// Block size along common dimension
        block_size_k: usize,
        /// Block size along dimension of rhs
        block_size_n: usize,
    },
    #[cfg(feature = "autotune")]
    /// Using autotune to chose the best kernel based on runtime information.
    #[default]
    Autotune,
}

#[cfg(not(feature = "autotune"))]
impl Default for MatmulStrategy {
    fn default() -> Self {
        MatmulStrategy::Tiling2d
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    strategy: MatmulStrategy,
) -> JitTensor<R, E, D> {
    match strategy {
        MatmulStrategy::Simple { grid_x, grid_y } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_simple(lhs, rhs, out, grid_x, grid_y)
        }
        MatmulStrategy::Tiling2d {
            grid_x,
            grid_y,
            block_size_m,
            block_size_n,
            block_size_k,
        } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d(
                lhs,
                rhs,
                out,
                grid_x,
                grid_y,
                block_size_m,
                block_size_n,
                block_size_k,
            )
        }
        MatmulStrategy::Tiling2dPadded {
            grid_x,
            grid_y,
            block_size_m,
            block_size_n,
            block_size_k,
        } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_padded(
                lhs,
                rhs,
                out,
                grid_x,
                grid_y,
                block_size_m,
                block_size_n,
                block_size_k,
            )
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => matmul_autotune(lhs, rhs),
    }
}

pub(crate) const B_M: usize = 64;
pub(crate) const B_N: usize = 64;
pub(crate) const B_K: usize = 32;
pub(crate) const WORKGROUP_SIZE: usize = 16;

// pub(super) fn make_workgroup<const D: usize>(output_shape: &Shape<D>) -> WorkGroup {
//     let num_blocks_x = f32::ceil(output_shape.dims[D - 2] as f32 / B_M as f32) as u32;
//     let num_blocks_y = f32::ceil(output_shape.dims[D - 1] as f32 / B_N as f32) as u32;
//     let mut num_blocks_z = 1;
//     for i in 0..D - 2 {
//         num_blocks_z *= output_shape.dims[i];
//     }

//     WorkGroup::new(num_blocks_x, num_blocks_y, num_blocks_z as u32)
// }

pub(super) fn make_info_handle<R: Runtime, E: JitElement, const D: usize>(
    lhs: &JitTensor<R, E, D>,
    rhs: &JitTensor<R, E, D>,
    output: &JitTensor<R, E, D>,
) -> Handle<R::Server> {
    let info = build_info(&[lhs, rhs, output]);
    rhs.client.create(bytemuck::cast_slice(&info))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_tiling_2d_launch<
    R: Runtime,
    E: JitElement,
    const D: usize,
    K: DynamicKernelSource + 'static,
>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    kernel: K,
) -> JitTensor<R, E, D> {
    todo!()
    // A tensor may need to be padded, in which case it will implicitly become contiguous
    // If not needed, it is only turned into contiguous if some batch dim has been swapped with row or col dim.
    // If batches were swapped among themselves, or if the last two dims are transposed, the underlying
    // kernel handles it without needing to turn it into contiguous.
    // let round_lhs = pad_round::<R, E, D>(lhs, B_M, B_K);
    // let lhs = match round_lhs {
    //     PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
    //         into_contiguous(tensor)
    //     }
    //     _ => round_lhs.into_tensor(),
    // };
    // let round_rhs = pad_round::<R, E, D>(rhs, B_K, B_N);
    // let rhs = match round_rhs {
    //     PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
    //         into_contiguous(tensor)
    //     }
    //     _ => round_rhs.into_tensor(),
    // };

    // let rounded_output_shape = shape_out(&lhs, &rhs);

    // let rounded_output = empty_device(
    //     rhs.client.clone(),
    //     rhs.device.clone(),
    //     rounded_output_shape.clone(),
    // );

    // let workgroup = make_workgroup(&rounded_output_shape);
    // let info_handle = make_info_handle(&lhs, &rhs, &rounded_output);

    // lhs.client.execute(
    //     Box::new(DynamicKernel::new(kernel, workgroup)),
    //     &[
    //         &lhs.handle,
    //         &rhs.handle,
    //         &rounded_output.handle,
    //         &info_handle,
    //     ],
    // );

    // crop(rounded_output, output)
}

pub(crate) fn launch_options<const D: usize>(
    lhs_shape: &Shape<D>,
    rhs_shape: &Shape<D>,
    output_shape: &Shape<D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> WorkGroup {
    let num_rows = lhs_shape.dims[D - 2];
    let num_cols = rhs_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32)
}
