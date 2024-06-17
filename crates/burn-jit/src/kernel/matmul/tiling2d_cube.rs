use crate::{fusion::kernel, kernel::into_contiguous, tensor::JitTensor, FloatElement, JitRuntime};
use burn_cube::prelude::*;

use super::{tiling2d_launch_options, tiling2d_shader::write_to_output, Tiling2dConfig};

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
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Shared memory size lhs: technically derivable from others, but needs comptime arithmetic
    pub sm_size_lhs: UInt,
    /// Shared memory size rhs: technically derivable from others, but needs comptime arithmetic
    pub sm_size_rhs: UInt,
}

impl CubeTiling2dConfig {
    fn new(config: Tiling2dConfig, k: usize) -> Self {
        let tile_size = config.tile_size_m;
        let sm_size_lhs = config.block_size_m * config.block_size_k * tile_size;
        let sm_size_rhs = config.block_size_k * config.block_size_n * tile_size;

        CubeTiling2dConfig {
            block_size_m: UInt::new(config.block_size_m as u32),
            block_size_k: UInt::new(config.block_size_k as u32),
            block_size_n: UInt::new(config.block_size_n as u32),
            unroll: config.unroll,
            check_k_bounds: k % config.block_size_k != 0,
            sm_size_lhs: UInt::new(sm_size_lhs as u32),
            sm_size_rhs: UInt::new(sm_size_rhs as u32),
        }
    }
}

#[derive(CubeType)]
struct Tiling2dState<F: Float> {
    pub n_loops: UInt,
    pub k: UInt,
    pub lhs: Tensor<F>,
    pub rhs: Tensor<F>,
    pub out: Tensor<F>,
    pub offset_lhs: UInt,
    pub offset_rhs: UInt,
    pub offset_output: UInt,
    pub row: UInt,
    pub col: UInt,
    pub dim_m: UInt,
    pub dim_k: UInt,
    pub dim_n: UInt,
    pub unit_col: UInt,
    pub unit_row: UInt,
    pub shared_lhs: SharedMemory<F>,
    pub shared_rhs: SharedMemory<F>,
    pub register_m: F,
    pub register_n: F,
    pub results: Array<F>,
    pub lhs_stride_col: UInt,
    pub lhs_stride_row: UInt,
    pub rhs_stride_col: UInt,
    pub rhs_stride_row: UInt,
    pub out_stride_row: UInt,
    pub out_stride_col: UInt,
}

#[cube]
fn gather_kernel_information<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) -> Tiling2dState<F> {
    // Config variables
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let sm_size_lhs = Comptime::map(config, |c| c.sm_size_lhs);
    let sm_size_rhs = Comptime::map(config, |c| c.sm_size_rhs);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    // Assumes lhs and rhs share vectorization factor
    let tile_size = Comptime::vectorization(lhs);

    // Topology info
    let n_threads_per_row = ((Comptime::runtime(block_size_n) - UInt::new(1))
        / Comptime::runtime(tile_size))
        + UInt::new(1);
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
    let skip_row = CUBE_POS_X * Comptime::runtime(block_size_m);
    let skip_col = CUBE_POS_Y * Comptime::runtime(block_size_n);

    // Position of the first element of the unit, relative to the cube
    let unit_row = (local_idx / n_threads_per_row) * Comptime::runtime(tile_size);
    let unit_col = (local_idx % n_threads_per_row) * Comptime::runtime(tile_size);

    // Position of the first element of the unit, in absolute (in one batch)
    let row = skip_row + unit_row;
    let col = skip_col + unit_col;

    // Batch offset for output
    let offset_output = dim_m * dim_n * batch;

    // Calculate offset for lhs and rhs, without regards to batches
    let mut offset_lhs = skip_row * lhs_stride_row;
    let mut offset_rhs = skip_col * rhs_stride_col;

    // Batch offset for lhs, rhs
    for b in range(0, rank - UInt::new(2), unroll) {
        let tmp = offset_output / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    let results = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    let shared_lhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_lhs), Comptime::get(tile_size));
    let shared_rhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_rhs), Comptime::get(tile_size));

    // Calculate exact number of loop iterations
    let mut n_loops = UInt::new(0); // TODO support more syntax
    if Comptime::get(check_k_bounds) {
        n_loops = UInt::cast_from(F::ceil(
            F::cast_from(dim_k) / F::cast_from(Comptime::runtime(block_size_k)),
        ));
    } else {
        n_loops = dim_k / Comptime::runtime(block_size_k);
    }

    // Dummy declarations
    let k = UInt::new(0);
    let register_m = F::new(0.);
    let register_n = F::new(0.);

    Tiling2dState {
        n_loops,
        k,
        lhs,
        rhs,
        out,
        offset_lhs,
        offset_rhs,
        offset_output,
        row,
        col,
        dim_m,
        dim_k,
        dim_n,
        unit_col,
        unit_row,
        shared_lhs,
        shared_rhs,
        register_m,
        register_n,
        results,
        lhs_stride_col,
        lhs_stride_row,
        rhs_stride_col,
        rhs_stride_row,
        out_stride_row,
        out_stride_col,
    }
}

#[cube]
fn load_shared_memory<F: Float>(
    kernel_state: Tiling2dState<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    load_tensor_with_checks(
        kernel_state.lhs,
        kernel_state.offset_lhs,
        kernel_state.shared_lhs,
        kernel_state.unit_col,
        kernel_state.unit_row,
        kernel_state.lhs_stride_col,
        kernel_state.lhs_stride_row,
        kernel_state.dim_m,
        kernel_state.row,
        kernel_state.k,
        kernel_state.dim_k,
        config,
        Comptime::new(true),
    );
    load_tensor_with_checks(
        kernel_state.rhs,
        kernel_state.offset_rhs,
        kernel_state.shared_rhs,
        kernel_state.unit_row,
        kernel_state.unit_col,
        kernel_state.rhs_stride_row,
        kernel_state.rhs_stride_col,
        kernel_state.dim_n,
        kernel_state.col,
        kernel_state.k,
        kernel_state.dim_k,
        config,
        Comptime::new(false),
    )
}

#[cube]
fn load_tensor_with_checks<F: Float>(
    input: Tensor<F>,
    input_offset: UInt,
    mut shared_memory: SharedMemory<F>,
    unit_idx_1: UInt,
    unit_idx_2: UInt,
    stride_1: UInt,
    stride_2: UInt,
    dim: UInt,
    pos_in_dim: UInt,
    k: UInt,
    dim_k: UInt,
    config: Comptime<CubeTiling2dConfig>,
    is_lhs: Comptime<bool>, // TODO support match enum
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::vectorization(input);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    let mut n_writes = Comptime::runtime(tile_size);
    if Comptime::get(check_k_bounds) {
        n_writes = UInt::min(dim - pos_in_dim, Comptime::runtime(tile_size));
    }

    // TODO we should avoid that if in no_check_bound version
    if n_writes >= UInt::new(1) {
        for j in range(0, Comptime::get(tile_size), unroll) {
            let current = unit_idx_1 + j;

            if current + k < dim_k {
                let mut sm_position = UInt::new(0); // TODO support let x = if... syntax
                if current < Comptime::runtime(block_size_k) {
                    if Comptime::get(is_lhs) {
                        sm_position = current
                            + unit_idx_2 / Comptime::runtime(tile_size)
                                * Comptime::runtime(block_size_k);
                    } else {
                        sm_position = (current * Comptime::runtime(block_size_n) + unit_idx_2)
                            / Comptime::runtime(tile_size)
                    }
                }

                let position_base = (k + current) * stride_1 + unit_idx_2 * stride_2 + input_offset;

                // TODO simplify when stride_2 is 1, so we can leverage already vectorized
                let mut array = Array::<F>::new(Comptime::get(tile_size));

                for i in range(0, n_writes, Comptime::new(false)) {
                    array[i] = input[position_base + i * stride_2];
                }
                // Pad with zeros
                if Comptime::get(check_k_bounds) {
                    for i in range(n_writes, Comptime::get(tile_size), Comptime::new(false)) {
                        array[i] = F::new(0.);
                    }
                }

                // TODO could tile_size be fetched from array length?
                shared_memory[sm_position] = array.to_vectorized(tile_size);
            }
        }
    }
}

#[cube]
fn computation_loop<F: Float>(
    kernel_state: Tiling2dState<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let unit_col = kernel_state.unit_col;
    let unit_row = kernel_state.unit_row;
    let shared_lhs = kernel_state.shared_lhs;
    let shared_rhs = kernel_state.shared_rhs;
    let mut register_m = kernel_state.register_m;
    let register_n = kernel_state.register_n;
    let results = kernel_state.results;
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let vectorization = Comptime::vectorization(kernel_state.lhs);

    for dot_index in range(0, Comptime::get(block_size_k), unroll) {
        register_m = shared_lhs[unit_row / Comptime::runtime(vectorization)
            * Comptime::runtime(block_size_k)
            + dot_index];
    }
}

#[cube(launch)]
pub fn tiling2d_matmul_kernel<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let kernel_state = gather_kernel_information::<F>(lhs, rhs, out, config);

    for i in range(0u32, kernel_state.n_loops, Comptime::new(false)) {
        let k = i * Comptime::runtime(block_size_k);

        load_shared_memory(kernel_state, config);

        sync_units();

        computation_loop(kernel_state);

        sync_units();
    }

    write_to_output(kernel_state);
}

/// Matrix multiplication using tiling 2d algorithm with
/// written in Cube
pub fn matmul_tiling_2d_cube<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    // Bound checks can be done comptime specifically for all dims
    // let bounds_check_required = check_bound_requirement(&lhs.shape, &rhs.shape, &config);
    let k = lhs.shape.dims[D - 1];

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
        CubeTiling2dConfig::new(config, k),
    );

    out
}
