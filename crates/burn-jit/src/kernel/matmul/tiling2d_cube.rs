use crate::{kernel::into_contiguous, tensor::JitTensor, FloatElement, JitRuntime};
use burn_cube::prelude::*;

use super::{tiling2d_launch_options, tiling2d_shader::gather_shader_information, Tiling2dConfig};

impl Init for CubeTiling2dConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Debug, Clone, Copy)]
/// Tiling 2D parameters
pub struct CubeTiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: UInt,
    /// Block size along common dimension
    pub block_size_k: UInt,
    /// Block size along dimension of rhs
    pub block_size_n: UInt,
    /// Loop unrolling
    pub unroll: bool,
}

impl From<Tiling2dConfig> for CubeTiling2dConfig {
    fn from(value: Tiling2dConfig) -> Self {
        // We ignore grid: it's CUBE_DIM
        // We ignore tile size -> they are the vectorization factor
        CubeTiling2dConfig {
            block_size_m: UInt::new(value.block_size_m as u32),
            block_size_k: UInt::new(value.block_size_k as u32),
            block_size_n: UInt::new(value.block_size_n as u32),
            unroll: value.unroll,
        }
    }
}

#[derive(CubeType)]
struct Tiling2dState {}

#[cube]
fn gather_kernel_information<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) -> Tiling2dState {
    // Config variables
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size_m = Comptime::vectorization(lhs);
    let tile_size_n = Comptime::vectorization(rhs);

    // Topology info
    let n_threads_per_row =
        ((Comptime::get(block_size_n) - UInt::new(1)) / Comptime::get(tile_size_n)) + UInt::new(1);
    let results_size = Comptime::get(tile_size_m) * Comptime::get(tile_size_n);
    let local_idx = UNIT_POS;
    let batch = ABSOLUTE_POS_Z;

    // Shapes
    let rank = out.rank();
    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_k = lhs.shape(rank - UInt::new(1));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Strides
    let lhs_stride_row = lhs.stride(rank - UInt::new(2));
    let lhs_stride_col = lhs.stride(rank - UInt::new(1));
    let rhs_stride_row = rhs.stride(rank - UInt::new(2));
    let rhs_stride_col = rhs.stride(rank - UInt::new(1));
    let out_stride_row = out.stride(rank - UInt::new(2));
    let out_stride_col = out.stride(rank - UInt::new(1));

    // Cube offset
    let skip_row = CUBE_POS_X * Comptime::get(block_size_m);
    let skip_col = CUBE_POS_Y * Comptime::get(block_size_n);

    // Position of the first element of the unit, relative to the cube
    let unit_row = (local_idx / n_threads_per_row) * Comptime::get(tile_size_m);
    let unit_col = (local_idx % n_threads_per_row) * Comptime::get(tile_size_n);

    // Position of the first element of the unit, in absolute (in one batch)
    let row = skip_row + unit_row;
    let col = skip_col + unit_col;

    // Batch offset for output
    let offset_output = dim_m * dim_n * batch;

    // Calculate offset for lhs and rhs, without regards to batches
    let mut offset_lhs = skip_row * lhs_stride_row;
    let mut offset_rhs = skip_col * rhs_stride_col;

    // Batch offset for lhs, rhs
    for b in range(0, rank - UInt::new(2)) {
        let tmp = offset_output / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    // TODO
    let results = Array::<F>::new(results_size);
}

#[cube(launch)]
pub fn tiling2d_matmul_kernel<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let kernel_state = gather_kernel_information(lhs, rhs, out, config);
    // let shader_state = gather_shader_information(scope, &self);

    // let block_size_k: Variable = self.config.block_size_k.into();
    // cpa!(
    //     scope,
    //     range(0u32, shader_state.n_loops).for_each(|i, scope| {
    //         // From 0 to K with steps block_size_k
    //         let k = shader_state.k;
    //         cpa!(scope, k = i * block_size_k);

    //         load_shared_memory(scope, &self, &shader_state);

    //         scope.register(Synchronization::SyncUnits);

    //         computation_loop(scope, &self, &shader_state);

    //         scope.register(Synchronization::SyncUnits);
    //     })
    // );

    // write_to_output(scope, &self, &shader_state);
}

/// Matrix multiplication using tiling 2d algorithm with
/// written in Cube
pub fn matmul_tiling_2d<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    // Bound checks can be done comptime specifically for all dims
    // let bounds_check_required = check_bound_requirement(&lhs.shape, &rhs.shape, &config);

    let client = lhs.client.clone();

    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    let cube_count = tiling2d_launch_options(&out.shape, config.clone());

    let settings = KernelSettings::default()
        .vectorize_input(0, 1)
        .vectorize_input(1, 1)
        .vectorize_output(0, 1);

    tiling2d_matmul_kernel_launch::<E::CubeElement, R>(
        client,
        cube_count,
        settings,
        TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        config.into(),
    );

    out
}
